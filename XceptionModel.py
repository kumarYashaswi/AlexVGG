import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3(**kwargs)


class Xception(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(Xception, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = depthwise3x3(3, 32)
        self.Conv2d_2a_3x3 = depthwise3x3(32, 32)
        self.Conv2d_2b_3x3 = depthwise3x3(32, 64)
        self.Conv2d_3b_5x5 = depthwise5x5(64, 80)
        self.Conv2d_4a_5x5 = depthwise5x5(80, 192)
        self.Mixed_5b = LayerA(192)
        self.Mixed_5c = LayerA(304)
        self.Mixed_5d = LayerA(304)
        self.Mixed_6a = LayerB(304)
        self.Mixed_6b = LayerC(352,mid_channels=128)
        self.Mixed_6c = LayerD(768)
        self.Mixed_6d = LayerE(832)
        self.Mixed_6e = LayerC(864)
        if aux_logits:
            self.AuxLogits = FinalLayer(864, num_classes)
        self.Mixed_7a = LayerD(864)
        self.Mixed_7b = LayerE(832)
        self.Mixed_7c = LayerE(864)
        self.fc = nn.Linear(864, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_5x5(x)
        x = self.Conv2d_4a_5x5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.max_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return x, aux
        return x


class LayerA(nn.Module):

    def __init__(self, in_channels):
        super(LayerA, self).__init__()
        
        self.depth3x3=depthwise3x3(in_channels,48)

        self.depth5x5 = depthwise5x5(in_channels,64)

        self.depth7x7 = depthwise7x7(in_channels,64)

        self.singlekernel = BasicConv1d(in_channels, 64)
        

    def forward(self, x):
        branch1 = self.depth3x3(x)
        branc2 = self.depth5x5(x)
        branch3 = self.depth7x7(x)
        branch4 = self.singlekernel(x)

        
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding="same")
        branch_pool = self.singlekernel(branch_pool)

        outputs = [branch1, branch2, branch3,branch4,branch_pool]
        k=torch.cat(outputs, 1)
        return k


class LayerB(nn.Module):

    def __init__(self, in_channels):
        super(LayeBA, self).__init__()
        
        self.depth3x3=depthwise3x3(in_channels,96)

        self.depth5x5 = depthwise5x5(in_channels,128)


        self.singlekernel = BasicConv1d(in_channels, 64)
        

    

    def forward(self, x):
        branch1 = self.depth3x3(x)
        branc2 = self.depth5x5(x)
        branch4 = self.singlekernel(x)

        
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding="same")
        branch_pool = self.singlekernel(branch_pool)

        outputs = [branch1, branch2,branch4,branch_pool]
        k=torch.cat(outputs, 1)
        return k



class LayerC(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super(InceptionC, self).__init__()
        self.branch1= BasicConv1d(in_channels, 192)

        c = mid_channels
        self.branch5x5_1 = BasicConv2d(in_channels, c, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(c, c, kernel_size=(1, 5), padding=(0, 3))
        self.branch5x5_3 = BasicConv2d(c, 192, kernel_size=(5, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))


    def forward(self, x):
        branch1 = self.branch1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch1(branch_pool)

        outputs = [branch1, branch5x5, branch7x7dbl, branch_pool]
        k=torch.cat(outputs, 1)
        return k


class LayerD(nn.Module):

    def __init__(self, in_channels):
        super(LayerD, self).__init__()
        
        self.depth3x3=depthwise3x3(in_channels,192)

        self.depth5x5 = depthwise5x5(in_channels,256)

        self.depth7x7 = depthwise7x7(in_channels,256)

        self.singlekernel = BasicConv1d(in_channels, 64)
        

    

    def forward(self, x):
        branch1 = self.depth3x3(x)
        branc2 = self.depth5x5(x)
        branch3 = self.depth7x7(x)
        branch4 = self.singlekernel(x)

        
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding="same")
        branch_pool = self.singlekernel(branch_pool)

        outputs = [branch1, branch2, branch3,branch4,branch_pool]
        k=torch.cat(outputs, 1)
        return k



class LayerE(nn.Module):

    def __init__(self, in_channels):
        super(LayerE, self).__init__()
        
        self.depth3x3=depthwise3x3(in_channels,192)

        self.depth5x5 = depthwise5x5(in_channels,256)

        self.PrevLayer = LayerB(in_channels)

        self.singlekernel = BasicConv1d(in_channels, 64)
        

    

    def forward(self, x):
        branch1 = self.depth3x3(x)
        branc2 = self.depth5x5(x)
        branch3 = self.PrevLayer(x)

        
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding="same")
        branch_pool = self.singlekernel(branch_pool)

        outputs = [branch1, branch2, branch3,branch_pool]
        k=torch.cat(outputs, 1)
        return k


class FinalLayer(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(FinalLayer, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return torch.cat(outputs, 1)

class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


    
class depthwise3x3(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(depthwise3x3, self).__init__()
        self.depth1 = nn.Conv2d(in_channels, in_channels,kernel_size=3,padding="same",**kwargs)
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.point1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.depth1(x)
        x = self.bn(x)
        x=  self.point2(x)
        x=self.bn1(x)
        return F.relu(x, inplace=True)

class depthwise5x5(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(depthwise5x5, self).__init__()
        self.depth1 = nn.Conv2d(in_channels, in_channels,kernel_size=5,padding="same",**kwargs)
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.point1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.depth1(x)
        x = self.bn(x)
        x=  self.point2(x)
        x=self.bn1(x)
        return F.relu(x, inplace=True)


class depthwise7x7(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(depthwise7x7, self).__init__()
        self.depth1 = nn.Conv2d(in_channels, in_channels,kernel_size=7,padding="same",**kwargs)
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.point1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.depth1(x)
        x = self.bn(x)
        x=  self.point2(x)
        x=self.bn1(x)
        return F.relu(x, inplace=True)                

             