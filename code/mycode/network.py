import torch
import torch.nn as nn


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class ECSAM_Module(nn.Module):


    def __init__(self, in_dim):
        super(ECSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.up = nn.Conv2d(in_dim,in_dim//2,1,)
        self.down = nn.Conv2d(in_dim//2,in_dim,1)
        self.conv = nn.Conv3d(1, 1, 5, 1, 2)
        self.sigmoid = nn.Sigmoid()
        self.gamma = 1
    def forward(self, x):
        out = self.up(x)
        m_batchsize, C, height, width = out.size()
        out = out.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        out = self.down(out)
        x = x * out + x
        return x

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network_'s weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Fusion(BaseNetwork):
    def __init__(self, dim, init_weights=True):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim,groups=dim, kernel_size=5, padding=2),
            nn.Conv2d(dim,dim,1),
            nn.InstanceNorm2d(dim , track_running_stats=False),
            nn.ReLU(True)
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        return self.fusion(x)

class Bridge(BaseNetwork):
    def __init__(self,dim, init_weights=True):
        super(Bridge, self).__init__()
        self.bridge =nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, 1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True)
        )
        if init_weights:
            self.init_weights()
    def forward(self,x):
        return self.bridge(x)


class Expand(BaseNetwork):
    def __init__(self,dim,init_weights=True):
        super(Expand, self).__init__()
        self.e1_1 = nn.Conv2d(dim,dim,3,2,1,groups=dim)
        self.e1_2 = nn.Conv2d(dim,dim,1)
        self.e2_1 = nn.Conv2d(dim, dim, 3, 2, 1,groups=dim)
        self.e2_2 = nn.Conv2d(dim, dim, 1)
        self.e3_1 = nn.Conv2d(dim,dim,5,2,2,groups=dim)
        self.e3_2 = nn.Conv2d(dim,dim,1)
        self.e4_1 = nn.Conv2d(dim,dim,7,2,3,groups=dim)
        self.e4_2 = nn.Conv2d(dim, dim, 1)

        if init_weights:
            self.init_weights()

    def forward(self,x):
        e1 = self.e1_1(x)
        e1 = self.e1_2(e1)
        e2 = self.e2_1(x)
        e2 = self.e2_2(e2)
        e3 = self.e3_1(x)
        e3 = self.e3_2(e3)
        e4 = self.e4_1(x)
        e4 = self.e4_2(e4)
        return torch.cat([e1,e2,e3,e4],1)

class GenerBlock(BaseNetwork):
    def __init__(self, dim,init_weights=True,type=None):
        super(GenerBlock, self).__init__()
        self.type = type
        self.con = nn.Sequential(
            Expand(dim),
            nn.InstanceNorm2d(dim*2,track_running_stats=False),
            nn.ReLU(True),
            nn.PixelShuffle(2),
            nn.Conv2d(dim,dim,3,1,1)
        )
        self.fusion = Fusion(dim)
        if init_weights:
            self.init_weights()
    def forward(self,x):
        out = self.con(x)
        out = out+self.fusion(x)
        return out

class GFgen(BaseNetwork):
    def __init__(self,dim,aphla):
        super(GFgen, self).__init__()
        self.alpha = aphla
        self.f1 = GenerBlock(dim)
        self.f2 = GenerBlock(dim)
        self.f3 = GenerBlock(dim)
        # self.f4 = GenerBlock(dim)

    def forward(self, x,y):
        f1 = self.f1(x+y*self.alpha)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        # f4 = self.f4(f3)
        return f3

class DFgen(BaseNetwork):
    def __init__(self,dim,init_weights=True):
        super(DFgen, self).__init__()
        self.f1 = GenerBlock(dim,type="D")
        self.f2 = GenerBlock(dim,type="D")
        self.f3 = GenerBlock(dim,type="D")
        self.f4 = GenerBlock(dim,type="D")

    def forward(self,x):
        f1 = self.f1(x)
        f2 = self.f2(f1+0.5*x)
        f3 = self.f3(f2+0.5*x+0.5*f1)
        f4 = self.f4(f3+0.5*f2+0.5*x+0.5*f1)
        return f4

class UP(BaseNetwork):
    def __init__(self, in_channel,init_weights=True):
        super(UP, self).__init__()

        self.ge = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=4,stride=2, padding=1),
            nn.InstanceNorm2d(in_channel//2, track_running_stats=False),
            nn.ReLU(True),
        )
        if init_weights:
            self.init_weights()

    def forward(self,x):
        return self.ge(x)

class Last(BaseNetwork):
    def __init__(self,init_weights=True):
        super(Last, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),

        )
        if init_weights:
            self.init_weights()

    def forward(self,x):
        return self.encode(x)

class Genetator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(Genetator, self).__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, groups=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, 1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, groups=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, 1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, groups=128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, 1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, groups=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, 1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, groups=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, 1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True),
        )

        self.middle4 = GFgen(512,1)
        self.middle3 = GFgen(256,0.5)
        self.middle2 = GFgen(128,0.2)

        self.up4 = UP(512)
        self.up3 = UP(256)
        self.up2= UP(128)
        self.atten = ECSAM_Module(64)
        self.df = DFgen(64)

        self.b1 = Bridge(256)
        self.b2 = Bridge(128)

        self.last =Last()


        if init_weights:
            self.init_weights()

    def forward(self, x,y=None):

        f1 = self.encode1(x) #256
        f2 = self.encode2(f1) #128
        f3 = self.encode3(f2) #64
        f4 = self.encode4(f3)  # 32

        o4 = self.middle4(f4,f4) #32
        o3 = self.up4(o4)
        o3 = self.middle3(o3, self.b1(f3))  # 64
        o2 = self.up3(o3) #128
        o2 = self.middle2(o2, self.b2(f2))  # 128
        o1 = self.up2(o2) #256
        o1 = self.atten(o1)
        o1 = self.df(o1)

        out = self.last(o1)
        x = (torch.tanh(out) + 1) / 2
        return x,o4,o3,o2
