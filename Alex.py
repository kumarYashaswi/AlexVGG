import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image, ImageOps, ImageEnhance

use_cuda = torch.cuda.is_available()

root = './data'

trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(227, interpolation=2),transforms.Normalize((0.5,), (1.0,)),transforms.Grayscale(num_output_channels=3)])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100
num_epochs=10
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, padding=2,stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding="same",stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding="same",stride=2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2,padding="same"))
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding="same",stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2))
       #to be edited
        self.fc1 = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 10)
            )
       
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),256*6*6)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
        
model = CNN()
#CUDA not available
'''
if use_cuda:
     model = model.cuda()
'''
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training the Model
for epoch in range(num_epochs):
    for i, (x, labels) in enumerate(train_loader):
        images = Variable(x)
        labels = Variable(labels)
        
        # Optimizing steps
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_set)//batch_size, loss.data[0]))
            
            
#Pretrained Modelling
            import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def alexnet(pretrained=False, **kwargs):
    model = CNN(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

model=alexnet()



# Testing the Model
model.eval()  
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
torch.save(model.state_dict(), 'AlexNet.pkl')
