import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import math
from PIL import Image, ImageOps, ImageEnhance

root = './data'
trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(224, interpolation=2),transforms.Normalize((0.5,), (1.0,)),transforms.Grayscale(num_output_channels=3)])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                shuffle=False)

__all__ = [
    'VGG','vgg19',
    'vgg19_bn']


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',   
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10,init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


architechture = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#VGG_19 without Batch Normalization
def vgg19(pretrained=True, **kwargs):
    
    model = VGG(make_layers(architechture['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

MODEL=vgg19()

#VGG_19 with Batch Normalization
def vgg19_bn(pretrained=True, **kwargs):
    
    model = VGG(make_layers(architechture['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

MODEL=vgg19_bn()

MODEL.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for j, (images, labels) in enumerate(test_loader):
        images = Variable(images)
        labels=Variable(labels)
        outputs = MODEL(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(MODEL.state_dict(), 'VGG_pretrain.pkl')





#Alternate Implementation

import torchvision.models as models
vgg19 = models.vgg19(pretrained=True)
    
model = vgg19

#CUDA not available for the system
#if use_cuda:
#    model = model.cuda()

root = './data'

trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(224, interpolation=2),transforms.Normalize((0.5,), (1.0,)),transforms.Grayscale(num_output_channels=3)])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans)


train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                shuffle=False)


# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for j, (images, labels) in enumerate(test_loader):
        images = Variable(images)
        labels=Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(model.state_dict(), 'VGG_pretrained2.pkl')













