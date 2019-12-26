# coding: utf-8
import os
import time
import torch
from torch import nn
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
testpath = '../data/test.txt'
batch_size = 150
num_worker = 8

num_class = 2


def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor

start = time.time()
####################### transformer defination, dataset reader and loader
preprocess = transforms.Compose([
    # transforms.RandomChoice([transforms.RandomHorizontalFlip(p=1),
    #                          transforms.RandomVerticalFlip(p=1)]), # randomly select one for process
    transforms.ToTensor(), # preprocess was operated on original image, will rewrite on previous transform.
    transforms.Normalize(normMean, normStd)])

testset = SCdataset(testpath, preprocess)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

net = models.resnet34(pretrained=False, num_classes=num_class).cuda()  # num_class is 10
net.load_state_dict(torch.load('../model/epoch_45.pkl')) # load the finetune weight parameters

correct = 0
predictions = np.array([])
real = np.array([])
score = np.array([])
net.eval()
with torch.no_grad():
    for i, (img, target) in tqdm(enumerate(testloader)):
        target = target.cuda().long()
        prob = F.softmax(net(img.cuda()), 1, _stacklevel=5)
        prediction = torch.argmax(prob, dim=1)
        correct += (prediction == target).sum().item()

        predictions = np.concatenate((predictions, prediction.cpu().numpy()), axis=0)
        real = np.concatenate((real, target.cpu().numpy()), axis=0)
        score = np.concatenate((score, np.array(prob[:, 1].cpu())), axis=0)

Acc = correct * 1.0 / testset.__len__()
print('---ACC{:.4f}---epoch{:.4f}---epoch{:.4f}---'.format(Acc, 1, 1))

ShowConfusionMatrix(real, predictions)
ROCAUC(real, score)
print(time.time()-start)
##
# nn.ModuleList
# nn.ReLU
