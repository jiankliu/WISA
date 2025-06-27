#-*- coding : utf-8-*-
# coding:unicode_escape

from einops import rearrange
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch import nn

class TcnResidualLayer(nn.Module):
    def __init__(self, in_c, out_c, dilated=1, k=3, s=1, p=1, store_features=False):
        super().__init__()
        self.tcn0 = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=dilated),
            nn.ReLU(),
        )
        self.tcn1 = nn.Sequential(
            nn.Conv1d(out_c, out_c, kernel_size=k, stride=s, padding=p, dilation=dilated),
        )
        self.relu = nn.ReLU(inplace=False)
        self.store_features = store_features
        self.features = {}

    def forward(self, x):
        residual = x
        out = self.tcn0(x)
        if self.store_features:
            self.features['after_tcn0'] = out
        out = self.tcn1(out)
        out = out + residual
        out = self.relu(out)
        return out

class Dwt1dModule_Tcnnew(nn.Module):
    def __init__(
        self,
        wvlname='db1',
        J=3,
        yl_size=14,
        yh_size=[26, 18, 14],
        ks = 3,
        store_features=False
    ):
        super().__init__()
        self.wvlname = wvlname
        self.J = J
        self.yl_num = yl_size
        self.yh_num = yh_size
        self.yh_blocks = nn.ModuleList()

        self.store_features = store_features
        self.features = {}

        for i in self.yh_num:
            self.yh_blocks.append(
                nn.Sequential(
                    TcnResidualLayer(1, 32, store_features=store_features, k=ks, p=ks//2),
                    nn.Conv1d(32, 1, kernel_size=ks, padding=ks//2, dilation=1),
                    nn.ReLU(),
                )
            )
        self.yl_block = nn.Sequential(
            TcnResidualLayer(1, 32, store_features=store_features, k=ks, p=ks//2),
            nn.Conv1d(32, 1, kernel_size=ks, padding=ks//2, dilation=1),
            nn.ReLU(),
        )
        self.dwt = DWT1DForward(wave=self.wvlname, J=self.J)
        self.idwt = DWT1DInverse(wave=self.wvlname)

    def forward(self, x):
        B, T, H = x.shape
        x_r = rearrange(x, 'b t h  -> b h t')
        x_r = rearrange(x_r, 'b h t -> (b h) 1 t')

        yl, yh = self.dwt(x_r)
        yl_out = self.yl_block(yl)
        yh_out = []
        for i, yhi in enumerate(yh):
            yhi_out = self.yh_blocks[i](yhi)
            yh_out.append(yhi_out)

        out = self.idwt((yl_out, yh_out))
        out = rearrange(out, '(b h) 1 t -> b h t', b=B, h=H)
        out = rearrange(out, 'b h t -> b t h')

        return out
        
class Dwt1dResnetX_TCN_WISA(nn.Module):
    def __init__(
            self,
            wvlname='db1',
            J=3,
            yl_size=14,
            yh_size=[26, 18, 14],
            norm=None,
            inc=28,
            nx=64,
            ny=64,
            ks=3,
            input_neuron=880,
            output_neuron=64 * 64,
            store_features=False
    ):
        super().__init__()
        self.wvl = Dwt1dModule_Tcnnew(wvlname, J, yl_size, yh_size, store_features=store_features, ks=ks)
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.nx = nx
        self.ny = ny
        self.inc = inc
        self.dense_decoder = nn.Sequential(
            nn.Linear(input_neuron, 512),
            nn.BatchNorm1d(inc),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_neuron),
            nn.Sigmoid(),
        )
        self.norm = norm
        self.en_conv = nn.Sequential(
            nn.Conv2d(inc if inc % 2 == 0 else inc + 1, 64, kernel_size=7,  bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, kernel_size=3,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(256, 128, kernel_size=3,  bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(128, 64, kernel_size=5,  bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(64, 1, kernel_size=7, bias=True),
        )
        self.store_features = store_features
        self.features = {}

    def forward(self, x):
        y = self.wvl(x)
        bc, ict, neu = y.shape
        y = self.dense_decoder(y)
        y = y.reshape(bc, ict, self.nx, self.ny)
        y = self.en_conv(y)
        out = self.de_conv(y)
        return out

