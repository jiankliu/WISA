import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import scipy.io
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 48

############################################
# 1. 定义 GaborConv2d 层：构造 Gabor 卷积核
############################################
class GaborConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, device="cuda", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)
        # 初始化 Gabor 参数
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels), device=device).type(torch.float32))
        )
        self.theta = nn.Parameter(
            (3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels), device=device).type(torch.float32)
        )
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels, device=device))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.tensor([self.kernel_size[0] / 2], device=device))[0]
        self.y0 = torch.ceil(torch.tensor([self.kernel_size[1] / 2], device=device))[0]
        self.device = device

    def forward(self, input_image):
        # 生成网格坐标
        y, x = torch.meshgrid(
            [torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0], device=self.device),
             torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1], device=self.device)]
        )
        weight = torch.empty(self.weight.shape, device=self.device, requires_grad=False)
        # 对每个输出通道和输入通道计算对应的 Gabor 卷积核
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

############################################
# 2. 定义 Generator 模型：从 spike 重构 (1, 90, 90) 图像
############################################
class Generator(nn.Module):
    def __init__(self, input_dim):
        """
        input_dim: spike 数据的维度
        模型先通过全连接层将 spike 映射到 90x90，然后经过卷积网络细化生成图像。
        """
        super(Generator, self).__init__()
        # 将 spike 数据映射为 90 x 90 个神经元
        self.fc = nn.Linear(33*90, 90 * 90)
        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()  # 输出映射到 [-1,1]

        # 用卷积网络对初步重建的图像进行细化
        self.gabor = GaborConv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, device=device)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x 的 shape: [BATCHSIZE, input_dim]
        # import pdb
        # pdb.set_trace()
        out = self.fc(x.view(x.size(0), -1))       # 映射到 [BATCHSIZE, 90*90]
        out = self.activation(out)
        out = out.view(x.size(0), 1, 90, 90)  # reshape 成 (1, 90, 90)
        # 使用 Gabor 卷积和卷积层进行细化
        out = self.gabor(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out

############################################
# 3. 定义自定义 Dataset：加载 spike 数据（1.mat）和标签图片（png），标签尺寸为 (1, 90, 90)
############################################
class SpikeImageDataset(Dataset):
    def __init__(self, spike_mat_path, image_folder, transform_img=None):
        """
        spike_mat_path: 存放 spike 数据的 .mat 文件路径，假设内部变量名为 'spk'
        image_folder: 存放 label 图片的文件夹，图片尺寸为 (1, 90, 90)
        transform_img: 对 label 图片的预处理
        """
        mat_data = scipy.io.loadmat(spike_mat_path)
        self.spike_data = mat_data['spk']  # 假设数据形状为 [num_samples, input_dim]
        # 获取文件夹内的所有 png 文件
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        self.transform_img = transform_img
        
        assert len(self.spike_data) == len(self.image_paths), \
            "Spike 数据样本数和图片数必须一致！"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 获取 spike 数据，并转换成 tensor
        spike = self.spike_data[idx]
        spike = torch.tensor(spike, dtype=torch.float32)
        # 读取 label 图片，并转换为灰度图（保证 shape 为 (1, 90, 90)）
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform_img is not None:
            img = self.transform_img(img)
        return spike, img

############################################
# 4. 主程序：数据加载、模型训练
############################################
# if __name__ == '__main__':
#     # 调整 input_dim，请根据实际 1.mat 内的 spike 特征维度修改（此处示例为 100 维）
#     input_dim = 100  
#     num_epochs = 20
#     learning_rate = 0.001

#     # 定义 label 图片的预处理：调整为 90x90，转换为 tensor，并归一化到 [-1,1]
#     transform_img = transforms.Compose([
#         transforms.Resize((90, 90)),
#         transforms.ToTensor(),       # 输出 [0,1]
#         transforms.Normalize(mean=[0.5], std=[0.5])  # 调整到 [-1,1]
#     ])

#     # 构建数据集和 DataLoader（请修改 'path/to/png_images' 为实际图片文件夹路径）
#     dataset = SpikeImageDataset(spike_mat_path='1.mat', image_folder='path/to/png_images', transform_img=transform_img)
#     dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    
#     # 初始化 Generator 模型
#     model = Generator(input_dim=input_dim).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()  # 示例中用 MSELoss，可根据需要进行修改

#     # 训练循环
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0
#         for spikes, labels in dataloader:
#             spikes = spikes.to(device)   # shape: [BATCHSIZE, input_dim]
#             labels = labels.to(device)   # shape: [BATCHSIZE, 1, 90, 90]
#             optimizer.zero_grad()
#             output = model(spikes)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         avg_loss = epoch_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

#     # 保存训练好的模型参数
#     torch.save(model.state_dict(), 'generator_model_90x90.pth')
