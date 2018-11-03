import math
import pdb
import torch
import torch.nn as nn
from torch.nn.functional import softmax, interpolate


#########################################
#                                       #
#   ResNet for DeepLab v3+ Backbone     #
#                                       #
#########################################

def conv1x1(in_planes, out_planes, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


def conv3x3(in_planes, out_planes, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model

def conv3x3_simple(in_planes, out_planes, stride=1, drop_rate=0):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


def atrous_conv(in_planes, out_planes, atrous_rate, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                  padding=atrous_rate, dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0):
        super(BasicBlock, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = conv3x3_simple(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_simple(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.dp(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0):
        super(Bottleneck, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dp(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.dp(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, drop_rate=0):
        super(ResNet, self).__init__()
        self.expansion = block.expansion
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_rate=drop_rate)

        self.low_conv = nn.Conv2d(64*block.expansion, 64, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.atr_conv1 = atrous_conv(256*block.expansion,128,1)
        self.atr_conv2 = atrous_conv(256*block.expansion,128,2)
        self.atr_conv3 = atrous_conv(256*block.expansion,128,4)
        self.atr_conv4 = atrous_conv(256*block.expansion,128,16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.Dropout2d(p=drop_rate)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate=drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_x = self.layer1(x)
        x = self.layer2(low_x)
        x = self.layer3(x)
        x1 = self.atr_conv1(x)
        x2 = self.atr_conv2(x)
        x3 = self.atr_conv3(x)
        x4 = self.atr_conv4(x)
        if self.expansion != 1:
            low_x = self.low_conv(low_x)
        return torch.cat((x1,x2,x3,x4), 1), low_x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet_test():
    net = resnet152()
    net = nn.DataParallel(net)
    net.cuda()

    x = torch.randn((1,1,1024,1024))
    y1, y2 = net(x)
    print(y1.size())
    print(y2.size())

#resnet_test()




#########################################
#                                       #
#           DeepLab V3 +                #
#                                       #
#########################################


class Encoder(nn.Module):

    def __init__(self, drop_rate=0):
        super(Encoder, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = conv1x1(512, 256)
        self.atr_conv1 = atrous_conv(512, 256, 6, drop_rate)
        self.atr_conv2 = atrous_conv(512, 256, 12, drop_rate)
        self.atr_conv3 = atrous_conv(512, 256, 18, drop_rate)
        self.avgpool = nn.AvgPool2d(2) #impossible due to 15/2 = 7.5
        self.conv2 = conv1x1(512, 256)
        self.conv_cat = conv1x1(1280, 512)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.atr_conv1(x)
        y3 = self.atr_conv2(x)
        y4 = self.atr_conv3(x)
        y5 = self.avgpool(x)
        y5 = self.conv2(y5)
        y5 = interpolate(y5, scale_factor=2, mode='bilinear', align_corners=True)

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.conv_cat(y)
        return interpolate(y, scale_factor=4, mode='bilinear', align_corners=True)


class Deeplab_V3_Plus(nn.Module):

    def __init__(self, class_num=2, drop_rate=0):
        super(Deeplab_V3_Plus, self).__init__()
        self.resnet = resnet50(drop_rate=drop_rate)
        self.conv0 = conv1x1(64, 256, drop_rate)
        self.encoder = Encoder(drop_rate)
        self.conv1 = conv3x3(768, 512, drop_rate)
        self.conv2 = conv3x3(512, 256, drop_rate)
        self.conv3 = conv3x3(256, class_num)
        self.conv4 = conv5x5(class_num, class_num)
        self.conv5 = conv5x5(class_num, class_num)

    def forward(self, x):
        y, low_y = self.resnet(x)
        low_y = self.conv0(low_y)
        y = self.encoder(y)
        y = torch.cat([low_y, y], dim=1)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = interpolate(y, scale_factor=4, mode='bilinear', align_corners=True)
        y = self.conv4(y)
        y = self.conv5(y)
        y = softmax(y, dim=1)
        return y


def deeplab_test():
    net = Deeplab_V3_Plus()
    net = nn.DataParallel(net)
    net.cuda()

    x = torch.randn((1,1,1024,1024))
    y = net(x)
    print(y.size())

#deeplab_test()
