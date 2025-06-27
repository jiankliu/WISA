import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward
from utils import calculate_psnr, calculate_ssim, mkdir
from dataset import DatasetWISA
from dwtnets import Dwt1dResnetX_TCN_WISA
import lpips
parser = argparse.ArgumentParser(description='WISA-retina')
parser.add_argument('-c', '--cuda', type=str, default='1', help='select gpu card')
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-hl', '--hl', type=int, default=512)
parser.add_argument('-pl', '--pl', type=int, default=64)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WISA')
parser.add_argument('-m', '--model_name', type=str, default='model_best')
parser.add_argument('-r', '--resume_from', type=str, default=None)
parser.add_argument('--dataroot', type=str, default=None)
parser.add_argument('--subdir', type=str, default=None)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

resume_folder = args.resume_from
model_name = args.model_name
batch_size = args.batch_size
learning_rate = 1e-4
train_epoch = args.epoch
dataroot = args.dataroot
subdir = args.subdir

opt = 'adam'
opt_param = "{\"beta1\":0.9,\"beta2\":0.99,\"weight_decay\":0}"

random_seed = True
manual_seed = 123

scheduler = "MultiStepLR"
scheduler_param = "{\"milestones\": [400, 600], \"gamma\": 0.2}"

wvlname = args.wvl
j = args.jlevels
ks = args.kernel_size
hl = args.hl
pl = args.pl

if_save_model = False
eval_freq = 1
checkpoints_folder = args.logpath + '-' + args.wvl + '-' + str(args.jlevels) + '-ks' + str(ks) + '-epoch'+str(train_epoch)

def progress_bar_time(total_time):
    hour = int(total_time) // 3600
    minu = (int(total_time) % 3600) // 60
    sec = int(total_time) % 60
    return '%d:%02d:%02d' % (hour, minu, sec)

def main():
    global batch_size, learning_rate, random_seed, manual_seed, opt, opt_param, if_save_model, checkpoints_folder
    mkdir(os.path.join('logs', checkpoints_folder))
    if random_seed:
        seed = np.random.randint(0, 10000)
    else:
        seed = manual_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt_param_dict = json.loads(opt_param)
    scheduler_param_dict = json.loads(scheduler_param)

    cfg = {}
    cfg['rootfolder'] = os.path.join(dataroot, 'train')
    cfg['spikefolder'] = subdir
    cfg['imagefolder'] = 'gt'+str(pl)
    cfg['H'] = 64
    cfg['W'] = 64
    cfg['C'] = 30
    train_set = DatasetWISA(cfg)

    cfg = {}
    cfg['rootfolder'] = os.path.join(dataroot, 'val')
    cfg['spikefolder'] = subdir
    cfg['imagefolder'] = 'gt'+str(pl)
    cfg['H'] = 64
    cfg['W'] = 64
    cfg['C'] = 30
    test_set = DatasetWISA(cfg)

    print('train_set len', train_set.__len__())
    print('test_set len', test_set.__len__())

    train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False)

    item0 = train_set[0]
    s = item0['spikes']
    im=item0['image']

    lpips_m=lpips.LPIPS(net='alex')
    lpips_model = torch.nn.DataParallel(lpips_m).cuda()

    T, spH= s.shape
    s = s[None, :, 0:1]
    dwt = DWT1DForward(wave=wvlname, J=j)
    
    s_r = rearrange(s, 'b t h-> b h t')
    s_r = rearrange(s_r, 'b h t -> (b h) 1 t')
    yl, yh = dwt(s_r)
    yl_size = yl.shape[-1]
    yh_size = [yhi.shape[-1] for yhi in yh]
    
    ch,imH,imW = im.shape

    if args.resume_from:
        print("loading model weights from ", resume_folder)
        model = torch.load(os.path.join(resume_folder, model_name+'.pt'))
    else:
        print("create new model")
        model = Dwt1dResnetX_TCN_WISA(inc=T, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size,
                                        norm=None, ks=ks, input_neuron=spH,
                                       output_neuron=imH * imW, nx=imH, ny=imW)
    model = torch.nn.DataParallel(model).cuda()
    
    # optimizer
    if opt.lower() == 'adam':
        assert ('beta1' in opt_param_dict.keys() and 'beta2' in opt_param_dict.keys() and 'weight_decay' in opt_param_dict.keys())
        betas = (opt_param_dict['beta1'], opt_param_dict['beta2'])
        del opt_param_dict['beta1']
        del opt_param_dict['beta2']
        opt_param_dict['betas'] = betas
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, **opt_param_dict)
    elif opt.lower() == 'sgd':
        assert ('momentum' in opt_param_dict.keys() and 'weight_decay' in opt_param_dict.keys())
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **opt_param_dict)
    else:
        raise ValueError()

    lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **scheduler_param_dict)

    for epoch in range(train_epoch+1):
        print('Epoch %d/%d ... ' % (epoch, train_epoch))
        model.train()
        total_time = 0

        for i, item in enumerate(train_data_loader):
            start_time = time.time()
            spikes = item['spikes'].cuda()
            image = item['image'].cuda()
            optimizer.zero_grad()
            pred = model(spikes)
            loss = F.mse_loss(image, pred)

            loss.backward()
            optimizer.step()

            elapse_time = time.time() - start_time
            total_time += elapse_time
            lr_list = lr_scheduler.get_last_lr()
            lr_str = ""
            for ilr in lr_list:
                lr_str += str(ilr) + ' '
            print('\r[training] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it] | LOSS: %.4f | LR: %s' % (
                float(i + 1) / int(len(train_data_loader)) * 100, i + 1, int(len(train_data_loader)),
                progress_bar_time(total_time),
                progress_bar_time(total_time / (i + 1) * int(len(train_data_loader))),
                total_time / (i + 1),
                loss.item(),
                lr_str), end='')

        lr_scheduler.step()
        print('')
        if epoch % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                sum_ssim = 0.0
                sum_psnr = 0.0
                sum_num = 0
                sum_lpips = 0.0
                total_time = 0
                for i, item in enumerate(test_data_loader):
                    start_time = time.time()
                    spikes = item['spikes'].cuda() 
                    image = item['image'].cuda()
                    pred = model(spikes)
                    lips = torch.squeeze(lpips_model(image, pred)).cpu().numpy()
                    prediction = pred[0].permute(1,2,0).cpu().numpy()
                    gt = image[0].permute(1,2,0).cpu().numpy()
                    psnr=calculate_psnr(gt * 255.0, prediction * 255.0)
                    ssim= calculate_ssim(gt * 255.0, prediction * 255.0)
                    sum_ssim += ssim
                    sum_psnr += psnr
                    sum_lpips += lips.mean()

                    sum_num += 1
                    elapse_time = time.time() - start_time
                    total_time += elapse_time
                    print('\r[evaluating] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it]' % (
                        float(i + 1) / int(len(test_data_loader)) * 100, i + 1, int(len(test_data_loader)),
                        progress_bar_time(total_time),
                        progress_bar_time(total_time / (i + 1) * int(len(test_data_loader))),
                        total_time / (i + 1)), end='')

                sum_psnr /= sum_num
                sum_ssim /= sum_num
                sum_lpips /= sum_num
            print('')
            print('\r[Evaluation Result] PSNR: %.3f | SSIM: %.3f | LPIPS: %.3f' % (sum_psnr, sum_ssim, sum_lpips))

if __name__ == '__main__':
    main()
