# import torch, torchvision
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.nn import init
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.optim as optim


################ HYPER PARAMETERS ###################

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args(args=[])
print(opt)

cuda = True if torch.cuda.is_available() else False


############### LOADING DATA ###############
transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_loader = DataLoader(datasets.MNIST('./mnist/', train=True, download=True,transform=transform),batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./mnist/', train=False, download=True,transform=transform),batch_size=256, shuffle=True)
    

    
#################### UTILITY METHODS ######################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.conv_1 = nn.Sequential(nn.Conv2d(1,32,5,1,0,True), nn.ReLU(), nn.Conv2d(32,32,5,1,0,True), nn.ReLU()) #28->24->20->10
        self.maxpool_1 = nn.MaxPool2d(2,2) 
        self.conv_2 = nn.Sequential(nn.Conv2d(32,64,5,1,0,True), nn.ReLU(), nn.Conv2d(64,64,5,1,0,True), nn.ReLU()) #10->6->2->1
        self.maxpool_2 = nn.MaxPool2d(2,2) 
        self.out = nn.Sequential(nn.Linear(64*(1)**2,512), nn.ReLU(), nn.Linear(512,10))
        
        
    def forward(self,x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = x.view(x.size(0),-1)
        x = self.out(x)
        return x
    
    
classifier = Classifier()
classifier.apply(weights_init_normal)
classifier.cuda()
classifier.train()

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0

for epoch in range (opt.epochs):
    print('Epoch',epoch,'---------------------------------------------------')
    total = 0
    correct = 0
    for iteration, sampled_batch in enumerate(train_loader):
        optimizer.zero_grad()
        img, label = sampled_batch
        out = classifier(img.cuda())[0]
        loss = criterion(out, label.cuda())
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(out.data,1)
        total += len(label)
        correct += (predicted == label.cuda()).sum().item()
        
        
        running_loss += loss.item()
        if iteration % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, iteration + 1, running_loss / 100))
            running_loss = 0.0
        
    train_acc = 100.*correct/total
    print('Train accuracy after epoch',epoch,train_acc)
    
        
    ######### TEST DATA ACCURACY ##########
    with torch.no_grad():
        total = 0
        correct = 0
        test_loss = 0
        for iter_, sampled_batch in enumerate(test_loader):
            img, label = sampled_batch
            out = classifier(img.cuda())[0]
            loss = criterion(out, label.cuda())
            
            _, predicted = torch.max(out.data,1)
            total += len(label)
            correct += (predicted == label.cuda()).sum().item()
            test_loss += loss.item()
            
        test_acc = 100.*correct/total
        test_loss = test_loss/total
        print('After epoch',epoch,'test acc %f test loss %f' % (test_acc,test_loss))
            
    if epoch % 5==0:
        checkpoint = {
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint,'classifier_50_dim.pth.tar')
        
    
print('Finished Training')