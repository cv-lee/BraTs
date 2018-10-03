import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pdb
import torchvision
import torch.tensor
from torch.nn.functional import softmax
from torchvision import datasets, transforms
from torch.autograd import Variable


def conv3x3(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.ReLU(),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.ReLU())


def upsample(in_c, out_c, bias=True):
	return nn.Sequential(
        #nn.ReflectionPad2d(1),
		nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=bias),
        nn.ReLU())


class UNet(nn.Module):
    def __init__(self, class_num=2, useBN=False):
        super(UNet, self).__init__()
        self.output_dim = class_num
        self.conv1 = conv3x3(1, 64, useBN=useBN)
        self.conv2 = conv3x3(64, 128, useBN=useBN)
        self.conv3 = conv3x3(128, 256, useBN=useBN)
        self.conv4 = conv3x3(256, 512, useBN=useBN)
        self.conv5 = conv3x3(512, 1024, useBN=useBN)

        self.conv4m = conv3x3(1024, 512, useBN=useBN)
        self.conv3m = conv3x3(512, 256, useBN=useBN)
        self.conv2m = conv3x3(256, 128, useBN=useBN)
        self.conv1m = conv3x3(128, 64, useBN=useBN)

        self.conv0  = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(64, self.output_dim, 3, 1, 0),
                                    nn.ReLU())
        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(1024, 512)
        self.upsample43 = upsample(512, 256)
        self.upsample32 = upsample(256, 128)
        self.upsample21 = upsample(128, 64)

		## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.normal_(m.weight.data, mean=0, std=0.01)

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(self.max_pool(output1))
        output3 = self.conv3(self.max_pool(output2))
        output4 = self.conv4(self.max_pool(output3))
        output5 = self.conv5(self.max_pool(output4))

        conv5m_out = torch.cat((self.upsample54(output5), output4), 1)
        conv4m_out = self.conv4m(conv5m_out)
        conv4m_out = torch.cat((self.upsample43(output4), output3), 1)
        conv3m_out = self.conv3m(conv4m_out)

        conv3m_out = torch.cat((self.upsample32(output3), output2), 1)
        conv2m_out = self.conv2m(conv3m_out)

        conv2m_out = torch.cat((self.upsample21(output2), output1), 1)
        conv1m_out = self.conv1m(conv2m_out)

        final = self.conv0(conv1m_out)
        final = softmax(final, dim=1)
        return final


def test():
    net = UNet(class_num=2)
    y = net(torch.randn(3,1,240,240))
    print(y.size())

#test()

