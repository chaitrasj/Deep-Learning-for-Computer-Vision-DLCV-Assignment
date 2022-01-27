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


################ HYPER PARAMETERS ###################

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--name", type=str, default='1_Model', help="Name of folder to save images, model and Tensorboard plot")
opt = parser.parse_args(args=[])
print(opt)

cuda = True if torch.cuda.is_available() else False


############### LOADING DATA ###############
transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_loader = DataLoader(datasets.MNIST('../mnist/', train=True, download=True,transform=transform),batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('../mnist/', train=False, download=True,transform=transform),batch_size=opt.batch_size, shuffle=True)
    
transform_save = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=32),
        transforms.ToTensor()
    ])

    
#################### UTILITY METHODS ######################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
############## BUILDING DISCRIMINATOR MODEL ####################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout(0), nn.BatchNorm2d(16,0.8))   
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout(0),nn.BatchNorm2d(32,0.8))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout(0),nn.BatchNorm2d(64,0.8))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout(0),nn.BatchNorm2d(128,0.8))
        self.out = nn.Sequential(nn.Linear(in_features=512, out_features=1, bias=True), nn.Sigmoid())

         
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.out(x)
        return x

    
############## BUILDING GENERATOR MODEL ####################

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        self.init_size = opt.img_size//4 # 7x7
        
        self.dense = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=128*(self.init_size**2), bias=True))
        self.conv_1 = nn.Sequential(nn.BatchNorm2d(128), nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True), nn.Tanh())

      
    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0),128 ,self.init_size, self.init_size)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


###### DEFINING THE MODELS ######
generator = Generator(opt.latent_dim)
discriminator = Discriminator()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

    
####### DEFINING THE OPTIMIZER ##########
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

####### DEFINING THE LOSS FUNCTION ########
adv_loss = torch.nn.BCELoss()


################## TRAINING LOOP ##################
if cuda:
    generator.cuda()
    discriminator.cuda()
    Tensor = torch.cuda.FloatTensor 
else:
    Tensor = torch.FloatTensor

counter = 0
model_path = os.path.join(opt.name,'Models')
images_path = os.path.join(opt.name,'Images')
tensorboard_path = os.path.join(opt.name,'Tensorboard')

if not os.path.exists(model_path):
    os.makedirs(model_path)
    print('Creating directory',model_path)
if not os.path.exists(images_path):
    os.makedirs(images_path) 
    print('Creating directory',images_path)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path) 
    print('Creating directory',tensorboard_path)

writer = SummaryWriter(log_dir = tensorboard_path)  
saving_epoch = 0

for epoch in range(opt.epochs):
    print('Epoch',epoch,'------------------------------')
    for iteration, sampled_batch in enumerate(train_loader):
        img, label = sampled_batch
        img = img.cuda()
        
        if cuda: 
            real_labels = torch.unsqueeze(torch.ones(len(img)),1).cuda()
            fake_labels = torch.unsqueeze(torch.zeros(len(img)),1).cuda()
        else:
            real_labels = torch.unsqueeze(torch.ones(len(img)),1)
            fake_labels = torch.unsqueeze(torch.zeros(len(img)),1)
        
        # TRAIN GENERATOR
        optimizer_gen.zero_grad()  
        
        z = Tensor(np.random.normal(size = (img.size(0),opt.latent_dim))) # Sample noise as input
        gen_samples = generator(z) # Generate fake samples
        
        gen_loss = adv_loss(discriminator(gen_samples), real_labels)
        gen_loss.backward()
        optimizer_gen.step()
         
        
        # TRAIN DISCRIMINATOR
        optimizer_dis.zero_grad()
        
        pred_real = discriminator(img)
        pred_fake = discriminator(gen_samples.detach())
        
        dis_loss_real = adv_loss(pred_real, real_labels)
        dis_loss_fake = adv_loss(pred_fake, fake_labels)
        dis_loss = (dis_loss_real + dis_loss_fake)/2
        
        dis_loss.backward()
        optimizer_dis.step()
        
        # Finding the accuracy of Discriminator on real and fake images
        pred_r = (pred_real>0.5).float()
        pred_f = (pred_fake>0.5).float()
        
        acc_real = 100.*((pred_r==real_labels).sum().item() / len(real_labels))
        acc_fake = 100.*((pred_f==fake_labels).sum().item() / len(fake_labels))

        counter += 1
        writer.add_scalar('Loss/Train: Gen loss ', gen_loss.item(), counter)
        writer.add_scalar('Loss/Train: Dis Loss Real ', dis_loss_real.item(), counter)
        writer.add_scalar('Loss/Train: Dis Loss Fake ', dis_loss_fake.item(), counter)
        writer.add_scalar('Accuracy/Train: Dis Acc Real ', acc_real, counter)
        writer.add_scalar('Accuracy/Train: Dis Acc Fake ', acc_fake, counter)
            
        
        # Printing loss values every 50 iterations
        if (counter % 50 ==0):            
            print('[Epoch %d/%d Iter %d/%d] Gen loss %f Dis loss %f Real Acc %2f Fake Acc %2f' %(epoch, opt.epochs, iteration, len(train_loader), gen_loss.item(), dis_loss.item(), acc_real, acc_fake))

   
    # Saving the model every 10 epochs.
    if (epoch%10 == 0):
        saving_epoch = epoch
        checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_gen': optimizer_gen.state_dict(),
        'optimizer_dis': optimizer_dis.state_dict(),
        }
        torch.save(checkpoint,model_path+'/_'+str(saving_epoch)+'.pth.tar')
        
        # Every 10 epochs, randomly generated 25 images are saved to keep a check on how the generated iages are getting better with training.
        x = torch.stack([transform_save(x_) for x_ in gen_samples[:25].cpu()])
        save_image(x, images_path+'/epoch_'+str(epoch)+'_%d.png' % (counter), nrow=5, normalize=True)