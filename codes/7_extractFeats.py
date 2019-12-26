# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
# from torch.autograd import Variable
from torchvision import models, transforms

from PIL import Image
import numpy as np
import cv2
import math
from glob import glob
import csv

def LoadNet(modelpath):
    net = models.resnet34(pretrained=False, num_classes=2)
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    net.cuda()
    return net

# hook the feature extractor
features_blobs = [0]
def hook_feature(module, input, output):
    #    features_blobs.append(output.data.cpu().numpy())
    features_blobs[0] = output.data.cpu().numpy()


## image preprocessing
preprocess = transforms.Compose([transforms.ToTensor(),  # preprocess was operated on original image, will rewrite on previous transform.
                                transforms.Normalize([0.787, 0.5723, 0.769], [0.1193, 0.1878, 0.0974])])


############################main function #####################################
def main(ImgList):
    f1 = open('/home/cyyan/projects/ccRCC/results/feats.csv', 'a')
    f2 = open('/home/cyyan/projects/ccRCC/results/prediction.csv', 'a')

    net = LoadNet('../model/epoch_45.pkl')  ## load the network
    net._modules.get('avgpool').register_forward_hook(hook_feature)  # get feature maps

    for num, IMG_URL in enumerate(ImgList):
        print(IMG_URL)
        torch_img = preprocess(Image.open(IMG_URL)).unsqueeze(0)
        prediction = torch.argmax(F.softmax(net(torch_img.cuda()), dim=1), dim=1).numpy()

        writer1 = csv.writer(f1)
        writer1.writerow(features_blobs[0].squeeze())
        writer2 = csv.writer(f2)
        writer2.writerow(prediction)

    f1.close()
    f2.close()


if __name__ == '__main__':
    imglist = []
    for line in open('/home/cyyan/projects/ccRCC/data/test.txt', 'r'):
        line = line.rstrip()
        imglist.append(line.split(' ')[0])

    main(imglist)
    # fil = open('/home/cyyan/projects/ccRCC/results/feats.csv', 'a')
    # writer = csv.writer(fil)
    # writer.writerow(features_blobs[0].squeeze())
    # fil.close()

