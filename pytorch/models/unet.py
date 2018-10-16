import pdb
import torch
import torch.nn as nn

from torch.nn.functional import softmax


def conv3x3(in_c, out_c, kernel_size=3, stride=1, padding=1,
            bias=True, useBN=False, drop_rate=0):
    if useBN:
        return nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=0, bias=bias),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU())


def upsample(in_c, out_c, bias=True, drop_rate=0):
	return nn.Sequential(
        #nn.ReflectionPad2d(1),
		nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=bias),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU())


class UNet(nn.Module):
    def __init__(self, in_channel=1, class_num=2, useBN=False, drop_rate=0):
        super(UNet, self).__init__()
        self.output_dim = class_num
        self.drop_rate = drop_rate
        self.conv1 = conv3x3(in_channel, 64, useBN=useBN, drop_rate=self.drop_rate)
        self.conv2 = conv3x3(64, 128, useBN=useBN, drop_rate=self.drop_rate)
        self.conv3 = conv3x3(128, 256, useBN=useBN, drop_rate=self.drop_rate)
        self.conv4 = conv3x3(256, 512, useBN=useBN, drop_rate=self.drop_rate)
        self.conv5 = conv3x3(512, 1024, useBN=useBN, drop_rate=self.drop_rate)

        self.conv4m = conv3x3(1024, 512, useBN=useBN, drop_rate=self.drop_rate)
        self.conv3m = conv3x3(512, 256, useBN=useBN, drop_rate=self.drop_rate)
        self.conv2m = conv3x3(256, 128, useBN=useBN, drop_rate=self.drop_rate)
        self.conv1m = conv3x3(128, 64, useBN=useBN, drop_rate=self.drop_rate)

        self.conv0  = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(64, self.output_dim, 3, 1, 0),
                                    nn.Dropout2d(p=self.drop_rate),
                                    nn.ReLU())
        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(1024, 512, drop_rate=self.drop_rate)
        self.upsample43 = upsample(512, 256, drop_rate=self.drop_rate)
        self.upsample32 = upsample(256, 128, drop_rate=self.drop_rate)
        self.upsample21 = upsample(128, 64, drop_rate=self.drop_rate)

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
