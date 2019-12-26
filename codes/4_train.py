# coding: utf-8
import os
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from _segclsDataReader import SCdataset
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from _utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(3)
np.random.seed(3)
torch.cuda.manual_seed(3)

#%% #######################hyparameter
normMean = [0.787,  0.5723, 0.769]
normStd = [0.1193, 0.1878, 0.0974]
trainpath = '../data/train.txt'
testpath = '../data/test.txt'
batch_size = 48
num_worker = 16

num_class = 2
learning_rate = 0.001
momentum = 0.9
lr_step = 40
epoches = 100

####################### transformer defination, dataset reader and loader
preprocess = transforms.Compose([
    # transforms.RandomChoice([transforms.RandomHorizontalFlip(p=1),
    #                          transforms.RandomVerticalFlip(p=1)]), # randomly select one for process
    transforms.ToTensor(), # preprocess was operated on original image, will rewrite on previous transform.
    transforms.Normalize(normMean, normStd)])

trainset = SCdataset(trainpath, preprocess)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
testset = SCdataset(testpath, preprocess)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

net = models.resnet34(pretrained=False, num_classes=num_class).cuda()  # num_class is 10
net.load_state_dict(torch.load('../model/epoch_31.pkl')) # load the finetune weight parameters

#net_state_dict = net.state_dict() # get the new network dict
#pretrained_dict = torch.load('/home/cyyan/.cache/torch/checkpoints/resnet34-333f7ec4.pth') # load the pretrained model
#pretrained_dict_new = {k: v for k, v in pretrained_dict.items() if k in net_state_dict and net_state_dict[k].size() == v.size()} #check the same key in dict.items
#net_state_dict.update(pretrained_dict_new) # update the new network dict by new dict in pretrained
#net.load_state_dict(net_state_dict) # load the finetune weight parameters

weights = torch.tensor([0.7, 0.3]).cuda()
if weights is None:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss(weight=weights).cuda()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=lr_step, gamma=0.3)


print('Start training...')
for epoch in range(epoches):
    start = time.time()
    net.train()

    losses = 0.0
    for i, (img,label) in enumerate(trainloader):
        img = img.cuda()
        label = label.cuda().long()

        output = net(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

        print('Iteration {:3d} loss {:.6f}'.format(i + 1, loss.item()))

    correct = 0
    net.eval()
    with torch.no_grad():
        for i, (img, target) in tqdm(enumerate(testloader)):
            target = target.cuda().long()
            prediction = torch.argmax(F.softmax(net(img.cuda()), 1, _stacklevel=5), dim=1)
            correct += (prediction == target).sum().item()
    Acc = correct * 1.0 / testset.__len__()

    print('---Epoch{:3d}---Time(s){:.2f}---Averageloss{:.4f}---'.format(epoch, time.time()-start, losses/(i+1)))
    #lr  = optimizer.param_groups[0]['lr']
    print('---ACC{:.4f}---epoch{:.4f}---epoch{:.4f}---'.format(Acc, epoch, epoch))
    scheduler.step()

    torch.save(net.state_dict(), '../model/epoch_'+str(epoch+32)+'.pkl')
    print('Model has been saved!')

print('Finished Training!')
