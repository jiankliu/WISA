# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward

from transform import Compose, RandomCrop, RandomRotationFlip
from dataset import DatasetREDS
#from dwtnets import Dwt1dResnetX_TCN
#from dwtnets import Dwt1dResnetX_TCN_VAE
from utils import calculate_psnr, calculate_ssim, mkdir

from dataset import DatasetFudan

# from dwtnets import Dwt1dResnetX_TCN_waveSID_slmd_mv1_32
from Spk2ImgNet.nets_mv01 import *

# from cae_net import End2EndTrain

import matplotlib.pyplot as plt
import pandas as pd

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-c', '--cuda', type=str, default='0', help='select gpu card')
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-e', '--epoch', type=int, default=300)
parser.add_argument('-n', '--spH', type=int, default=90)
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('-r', '--resume_from', type=str, default=None)
parser.add_argument('--dataroot', type=str, default='../movie01/old')
#parser.add_argument('--Class', type=str, default='9')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

resume_folder = args.resume_from
batch_size = args.batch_size
learning_rate = 1e-4
train_epoch = args.epoch
dataroot = args.dataroot

opt = 'adam'
opt_param = "{\"beta1\":0.9,\"beta2\":0.99,\"weight_decay\":0}"

random_seed = True
manual_seed = 123

scheduler = "MultiStepLR"
scheduler_param = "{\"milestones\": [400, 600], \"gamma\": 0.2}"

spH=args.spH
wvlname = args.wvl
j = args.jlevels
ks = args.kernel_size
#Class = args.Class

if_save_model = False
eval_freq = 1
checkpoints_folder = args.logpath + '-' + args.wvl + '-' + str(args.jlevels) + '-' + 'ks' + str(ks)


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

    #spH = 842
    spW = 1
    cfg = {}
    cfg['rootfolder'] = os.path.join(dataroot, 'train')
    cfg['spikefolder'] =  'input'
    cfg['imagefolder'] = 'gt'
    cfg['H'] = 90
    cfg['W'] = 90
    cfg['C'] = 1
    train_set = DatasetFudan(cfg)

    cfg = {}
#     cfg['rootfolder'] = os.path.join(dataroot, 'train')
    cfg['rootfolder'] = os.path.join(dataroot, 'val')
    #cfg['spikefolder'] = 'input' + '/' + Class
    cfg['spikefolder'] = 'input'
    cfg['imagefolder'] = 'gt'
    cfg['H'] = 90
    cfg['W'] = 90
    cfg['C'] = 1
    test_set = DatasetFudan(cfg)

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
        drop_last=True)

    print(train_data_loader)
    print(test_data_loader)

    item0 = train_set[0]
    s = item0['spikes']   # s.shape = [45, 64]
    im=item0['image']
    print("spikes",s.shape)
    print("image",im.shape)
    s = s[None, :, 0:1]   # 取s中的45个T或者说45个神经元一次实验的数据  s.shape = [1,45,1]
    import pdb
#     pdb.set_trace()
    dwt = DWT1DForward(wave=wvlname, J=j)
    B, T, H= s.shape
    s_r = rearrange(s, 'b t h-> b h t')     # s_r.shape = [1,1,45]
    s_r = rearrange(s_r, 'b h t -> (b h) 1 t')   # s_r.shape = [1,1,45]
    yl, yh = dwt(s_r)
    yl_size = yl.shape[-1]
    yh_size = [yhi.shape[-1] for yhi in yh]
    print(yh_size)
    print(yl_size)
    ims=item0['image']
    ch,imH,imW = ims.shape
    print(ims.shape)
    import pdb
    #pdb.set_trace()
    # img=ims[0,:,:]
    # print(img.shape)
    # temp=img.reshape(-1,16*17)
    # img1=temp.reshape(16,17)
    # #img=np.squeeze(img1)
    # fig = plt.figure('show picture')
    # ax = fig.add_subplot(111)
    # ax.imshow(img)
    # ax.set_title("hei,i'am the title")
    # plt.axis('off')
    # plt.show()
    #model = Dwt1dResnetX_TCN_FudanEphy(inc=41, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks, input_neuron = H*W,hidden1_neuron = 20000,hidden2_neuron =18000,output_neuron = imH*imW,nx=imH,ny=imW)
    #model = Dwt1dResnetX_TCN_FudanEphy(inc=41, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks)
    #model = Dwt1dResnetX_TCN_VAE(inc=41, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks)
    #model = Dwt1dResnetX_TCN_SID(inc=T, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size,
                                       # num_residual_blocks=3, norm=None, ks=ks, input_neuron=spH,
                                       # output_neuron=imH * imW, nx=imH,
                                       # ny=imW)
    
    # model = Dwt1dResnetX_TCN_waveSID_slmd_mv1_32(inc=T, wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size,
    #                                    num_residual_blocks=3, norm=None, ks=ks, input_neuron=spH,
    #                                    output_neuron=imH * imW, nx=imH,
    #                                    ny=imW)
    model = SpikeNet(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)

    ## cae_net
#     cae_model = End2EndTrain(256)
    criterion = nn.L1Loss(size_average=True)
    criterion = criterion.cuda()
    print(model)
    if args.resume_from:
        print("loading model weights from ", resume_folder)
        saved_state_dict = torch.load(os.path.join(resume_folder, 'model_best' + '.pt'))
        model.load_state_dict(saved_state_dict.module.state_dict())
        print("Weighted loaded.")

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
    best_psnr, best_ssim = 0.0, 0.0
    
    ssim_list = []
    psnr_list = []
    loss_list = []
    epoch_list = []
    for epoch in range(train_epoch+1):
        print('Epoch %d/%d ... ' % (epoch, train_epoch))

        model.train()
        total_time = 0
        f = open(os.path.join('logs', checkpoints_folder, 'log.txt'), "a")
        for i, item in enumerate(train_data_loader):
        
            start_time = time.time()

            spikes = item['spikes'].cuda()
            image = item['image'].cuda()
            optimizer.zero_grad()
            #print("spikes",spikes.shape)
            #print("image",image.shape)
            # import pdb
            # pdb.set_trace()
            # pred = model(spikes)
            pred, est0, est1, est2, est3, est4 = model(spikes)
            #print("pred.shape",pred.shape)
            est0 = est0 / 0.6
            est1 = est1 / 0.6
            est2 = est2 / 0.6
            est3 = est3 / 0.6
            est4 = est4 / 0.6
            pred = pred / 0.6
            loss = criterion(image, pred)
            # import pdb
            # pdb.set_trace()
            # for slice_id in range(4):
            #     loss = loss + 0.02 * (
            #         criterion(image[:, :, :, :], est0[:, slice_id : slice_id + 1, :, :])
            #         + criterion(
            #             image[:, :, :, :], est1[:, slice_id : slice_id + 1, :, :]
            #         )
            #         + criterion(
            #             image[:, :, :, :], est2[:, slice_id : slice_id + 1, :, :]
            #         )
            #         + criterion(
            #             image[:, :, :, :], est3[:, slice_id : slice_id + 1, :, :]
            #         )
            #         + criterion(
            #             image[:, :, :, :], est4[:, slice_id : slice_id + 1, :, :]
            #         )
            #     )

            #loss = F.l1_loss(image, pred)
            # loss = F.mse_loss(image, pred)
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
            f.write('[training] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it] | LOSS: %.4f | LR: %s\n' % (
                float(i + 1) / int(len(train_data_loader)) * 100, i + 1, int(len(train_data_loader)),
                progress_bar_time(total_time),
                progress_bar_time(total_time / (i + 1) * int(len(train_data_loader))),
                total_time / (i + 1),
                loss.item(),
                lr_str))
            if i == 0:
                loss_list.append(loss.detach().cpu())

        lr_scheduler.step()

        print('')
        if epoch % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                sum_ssim = 0.0
                sum_psnr = 0.0
                sum_num = 0
                total_time = 0
                for i, item in enumerate(test_data_loader):
                    start_time = time.time()

                    spikes = item['spikes'].cuda()
                    image = item['image'].cuda()
                    #print("eval_spikes",spikes.shape)
                    # import pdb
                    # pdb.set_trace()
                    pred, est0, est1, est2, est3, est4 = model(spikes)
                    
                    prediction = pred[0].permute(1,2,0).cpu().numpy()
                    gt = image[0].permute(1,2,0).cpu().numpy()
                    #print("eval_gt",gt.shape)

                    sum_ssim += calculate_ssim(gt * 255.0, prediction * 255.0)
                    sum_psnr += calculate_psnr(gt * 255.0, prediction * 255.0)
                    sum_num += 1
                    elapse_time = time.time() - start_time
                    total_time += elapse_time
                    print('\r[evaluating] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it]' % (
                        float(i + 1) / int(len(test_data_loader)) * 100, i + 1, int(len(test_data_loader)),
                        progress_bar_time(total_time),
                        progress_bar_time(total_time / (i + 1) * int(len(test_data_loader))),
                        total_time / (i + 1)), end='')
                    f.write('[evaluating] %3.2f%% | %6d/%6d [%s<%s, %.2fs/it]\n' % (
                    float(i + 1) / int(len(test_data_loader)) * 100, i + 1, int(len(test_data_loader)),
                    progress_bar_time(total_time),
                    progress_bar_time(total_time / (i + 1) * int(len(test_data_loader))),
                    total_time / (i + 1)))
                sum_psnr /= sum_num
                sum_ssim /= sum_num
                
                ssim_list.append(sum_ssim)
                psnr_list.append(sum_psnr)
                epoch_list.append(epoch)

            print('')
            print('\r[Evaluation Result] PSNR: %.3f | SSIM: %.3f' % (sum_psnr, sum_ssim))
            f.write('[Evaluation Result] PSNR: %.3f | SSIM: %.3f\n' % (sum_psnr, sum_ssim))

        if if_save_model and epoch % eval_freq == 0:
            print('saving net...')
            torch.save(model, os.path.join('logs', checkpoints_folder) + '/model'+ 'movie01_1000_1' + '_epoch%d.pt' % epoch)
            print('saved')

        if sum_psnr > best_psnr or sum_ssim > best_ssim:
            best_psnr = sum_psnr
            best_ssim = sum_ssim
            print('saving best net...')
            torch.save(model, os.path.join('logs', checkpoints_folder) + '_model_' + 'movie01_1000_1' + '.pt')
            print('saved')

        f.close()


    # 假设 epoch_list, ssim_list, psnr_list, loss_list 已经定义

    #plt.rcParams.update({'font.size': 12, 'text.usetex': True}) # 启用 LaTeX 和调整字体大小

    # SSIM 曲线
    plt.figure(figsize=(8, 6), dpi=300) # 调整图像大小和分辨率
    plt.plot(epoch_list, ssim_list, color='navy', linestyle='-', marker='o', lw=2) # 使用深蓝色
    plt.title("Evaluation Curves of SSIM", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("SSIM", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # 添加细网格线
    plt.legend(["Evaling SSIM"], loc='best')
    plt.tight_layout()
    #plt.savefig("/home/pengjing/salamanda/img/new_model/waveletTCNN_movie03_ssim.jpg", dpi=300)

    # PSNR 曲线
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epoch_list, psnr_list, color='darkgreen', linestyle='-', marker='x', lw=2) # 使用墨绿色
    plt.title("Evaluation Curves of PSNR", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("PSNR", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(["Evaling PSNR"], loc='best')
    plt.tight_layout()
    #plt.savefig("/home/pengjing/salamanda/img/new_model/waveletTCNN_movie03_psnr.jpg.jpg", dpi=300)

    # Loss 曲线
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epoch_list, loss_list, color='maroon', linestyle='-', marker='s', lw=2) # 使用深红色
    plt.title("Evaluation Curves of Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(["Evaling Loss"], loc='best')
    plt.tight_layout()
    #plt.savefig('/home/pengjing/salamanda/img/new_model/waveletTCNN_movie03_loss.jpg.jpg', dpi=300)

    results_dict = {
    'Epoch': epoch_list,
    'SSIM': ssim_list,
    'PSNR': psnr_list,
    'Loss': loss_list
    }

    # Create a DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    # Save the DataFrame to a CSV file
    results_filename = './spk2imagenet_movie03_250.csv' ###### 需要根据任务修改
    results_df.to_csv(results_filename, index=False)

    print(f'Results saved to {results_filename}')

    """ plt.figure()
    train_ssim = plt.plot(epoch_list, ssim_list, 'g', lw=2)#lw为曲线宽度
    
    plt.title("evaling curves of ssim")
    plt.xlabel("epoch")
    plt.ylabel("ssim")
    plt.legend(["evaling_ssim"])
    plt.savefig(("/home/pengjing/Documents/code/salamanda/img/waveletCNN_movie03_ssim" + ".jpg"))
    
    plt.figure()
    train_ssim = plt.plot(epoch_list, psnr_list, 'g', lw=2)#lw为曲线宽度
    
    plt.title("evaling curves of psnr")
    plt.xlabel("epoch")
    plt.ylabel("psnr")
    plt.legend(["evaling_psnr"])
    plt.savefig(("/home/pengjing/Documents/code/salamanda/img/waveletCNN_movie03_psnr" + ".jpg"))
    
    plt.figure()
    train_ssim = plt.plot(epoch_list, loss_list, 'g', lw=2)#lw为曲线宽度
    
    plt.title("evaling curves of loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["evaling_loss"])
    plt.savefig(('/home/pengjing/Documents/code/salamanda/img/waveletCNN_movie03_loss' + '.jpg')) """
    
    
    # visualization the reconstructed images
    # X_reconstructed_mu = prediction
    # n = 10
    # for j in range(1):
    #     plt.figure(figsize=(12, 2))
    #     for i in range(n):
    #         # display original images
    #         ax = plt.subplot(2, n, i +j*n*2 + 1)
    #         plt.imshow(np.rot90(np.fliplr(gt[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         # display reconstructed images
    #         ax = plt.subplot(2, n, i + n + j*n*2 + 1)
    #         #plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
    #         plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #
    #     plt.show()
    #     plt.savefig('e2eRec_spk.png', dpi=300)


if __name__ == '__main__':
    main()
