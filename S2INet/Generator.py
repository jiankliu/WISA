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

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 48

######################################
# 1. 定义 GaborConv2d 层
######################################
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
        self.psi = nn.Parameter(
            3.14 * torch.rand(out_channels, in_channels, device=device)
        )
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.tensor([self.kernel_size[0] / 2], device=device))[0]
        self.y0 = torch.ceil(torch.tensor([self.kernel_size[1] / 2], device=device))[0]
        self.device = device

    def forward(self, input_image):
        # 生成网格坐标（注意：meshgrid 默认按照行列顺序）
        y, x = torch.meshgrid(
            [torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0], device=self.device),
             torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1], device=self.device)]
        )
        weight = torch.empty(self.weight.shape, device=self.device, requires_grad=False)
        # 对每个输出通道和输入通道生成对应的 Gabor 卷积核
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)

                # 坐标旋转
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                # 计算 Gabor 函数
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                # 归一化
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

######################################
# 2. 定义 Generator 模型（图像重建模型）
######################################
class Generator(nn.Module):
    def __init__(self, input_dim):
        """
        input_dim: spike 数据全连接层输入的维度
        假设经过全连接层将 spike 转换为32×32的特征图
        """
        super(Generator, self).__init__()
        self.fc_layer1 = nn.Linear(input_dim, 32 * 32)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.LeakyReLU()
        self.activation_out = nn.Tanh()  # 输出使用 Tanh，将数据映射到 [-1, 1]
        
        # 编码部分（下采样）
        self.h1 = GaborConv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, device=device)
        self.b1 = nn.BatchNorm2d(128)
        self.h2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(256)
        self.h3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.b3 = nn.BatchNorm2d(512)
        self.h4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.b4 = nn.BatchNorm2d(1024)
        
        # 解码部分（上采样）
        self.h5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.b5 = nn.BatchNorm2d(512)
        self.h6 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.b6 = nn.BatchNorm2d(256)
        self.h7 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.b7 = nn.BatchNorm2d(128)
        self.h8 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        self.b8 = nn.BatchNorm2d(1)
        
        # 参数初始化（可选）
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, c):
        """
        假设输入 c 的 shape 为 [BATCHSIZE, input_dim]
        经全连接层输出为 [BATCHSIZE, 32*32]，再 reshape 成 [BATCHSIZE, 1, 32, 32]
        """
        fc1 = self.fc_layer1(c)
        fc1 = self.activation(fc1)
        x = fc1.reshape(c.shape[0], 1, 32, 32)
        x = self.h1(x)
        x = self.b1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h2(x)
        x = self.b2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h3(x)
        x = self.b3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h4(x)
        x = self.b4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h5(x)
        x = self.b5(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h6(x)
        x = self.b6(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h7(x)
        x = self.b7(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.h8(x)
        x = self.b8(x)
        # 最后用激活函数映射到输出范围，比如 Tanh
        out = self.activation_out(x)
        return out

######################################
# 3. 自定义 Dataset（加载 spike 数据与 png 图像）
######################################
class SpikeImageDataset(Dataset):
    def __init__(self, spike_mat_path, image_folder, transform_img=None):
        """
        spike_mat_path: spike 数据 .mat 文件路径（假设内部变量名为 'spk'）
        image_folder: 存放目标图片的文件夹（.png 格式）
        transform_img: 针对图片的预处理操作
        """
        # 加载 .mat 文件（请根据实际变量名修改 'spk'）
        mat_data = scipy.io.loadmat(spike_mat_path)
        self.spike_data = mat_data['spk']  # 假设数据保存为 [num_samples, feature_dim]
        # 列出所有png图片
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        self.transform_img = transform_img

        assert len(self.spike_data) == len(self.image_paths), \
            "Spike 数据与图片数量必须一致！"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 获取 spike 数据，转换为 tensor
        spike = self.spike_data[idx]
        spike = torch.tensor(spike, dtype=torch.float32)
        # 若有需要，请在此处对 spike 做 reshape
        # 假设 spike 原始形状为 [input_dim]
        # 注意：Generator 中全连接层输入要求为 [BATCHSIZE, input_dim]
        
        # 读取对应图片
        img = Image.open(self.image_paths[idx]).convert('L')  # 灰度图
        if self.transform_img is not None:
            img = self.transform_img(img)
        # img 的形状预期为 [1, 32, 32]
        return spike, img

######################################
# 4. 主函数：数据加载和训练示例
######################################
if __name__ == '__main__':
    # 假设 spike 数据的每个样本为一个1维向量，维度为 input_dim
    # 请根据实际数据修改 input_dim
    input_dim = 100  # 示例：100维
    num_epochs = 10
    learning_rate = 0.001

    # 图像预处理（调整为32×32，并转换到 [-1, 1] 区间，如果使用 Tanh 激活）
    transform_img = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # 转换为 [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 调整为 [-1, 1]
    ])

    # 创建数据集（请调整文件路径为实际路径）
    dataset = SpikeImageDataset(spike_mat_path='1.mat', image_folder='path/to/png_images', transform_img=transform_img)
    dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)

    # 初始化模型
    model = Generator(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 示例中使用 MSELoss，实际可根据论文使用加权loss

    # 训练循环示例
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for spike_batch, img_target in dataloader:
            # 将数据转到 device
            spike_batch = spike_batch.to(device)
            img_target = img_target.to(device)
            
            # spike_batch 的 shape 为 [BATCHSIZE, input_dim]，请保证与模型要求一致
            optimizer.zero_grad()
            output = model(spike_batch)
            # output 的 shape 应为 [BATCHSIZE, 1, 32, 32] 与 img_target 匹配
            loss = criterion(output, img_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # 保存模型（可选）
    torch.save(model.state_dict(), 'generator_model.pth')
