# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:20:58 2018
@author: yann
Changing from github's CAM project. 

a possible way to run this script:
python cam_function.py --imgsize 200 
--modelpath '/home/yann/Projects/G3G4/model/resnet_params_72.pkl' 
--savepath '/home/yann/Projects/G3G4/DATA/' 
--imgpath '/home/yann/Projects/G3G4/DATA/11967_71.png'
"""

# simple implementation of CAM in PyTorch for the networks 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms

from PIL import Image
import numpy as np
import argparse
import cv2
import math
import os

## imgsize = 512 '/media/yann/FILE/Kidney/Yan/ResNet/model/resnet_params_25.pkl'
def LoadNet(imgsize, modelpath):
    net = models.resnet34(pretrained = False, num_classes= 2)
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
normMean = [0.787,  0.5723, 0.769]
normStd = [0.1193, 0.1878, 0.0974]
preprocess = transforms.Compose([
    # transforms.RandomChoice([transforms.RandomHorizontalFlip(p=1),
    #                          transforms.RandomVerticalFlip(p=1)]), # randomly select one for process
    transforms.ToTensor(), # preprocess was operated on original image, will rewrite on previous transform.
    transforms.Normalize(normMean, normStd)])

  
# get the softmax weight after the average pooling process
def GetWeightSoftmax(network, prediction, imgsize):
    params = list(network.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    weight = weight_softmax[prediction, :]
    return weight


# generate the class activation maps upsample to 512*512
def ReturnCAM(feature_conv, weight_softmax, input_h, input_w):
    size_upsample = (input_h, input_w)
    _, _, nc, h, w = np.shape(feature_conv) 
    cam = np.dot(weight_softmax, np.reshape(feature_conv, (nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam


##read the 'CAM' and overlap and save the image to 'savepath'
#savepath = '/media/yann/FILE/Kidney/Yan/ResNet/result/1119_visualization/'
def SaveImage(IMG_URL, CAM, savepath, savename):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
#    index = (CAM < 128) # reserve the region which value >128
    img = cv2.imread(IMG_URL, 1)
#    for channel in range(3):
#        c = img[:, :, channel]
#        c[index] = 0
#        img[:,:,channel] = c
        
    cv2.imwrite(savepath + savename + '.png', img)
        
    
    attentionmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
#    cv2.imwrite(savepath + savename + '_attentionmap.png', attentionmap)
       
    img[:,:,1] = img[:,:,2]
    img[:,:,0] = img[:,:,2]
    overlap = attentionmap * 0.5 + img * 0.5
    cv2.imwrite(savepath + savename + '_overlap.png', overlap) 
        

#### get args
def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-M','--modelpath', type=str,
                        default='../model/epoch_45.pkl',
	                    help='the network model path')

    parser.add_argument('--multi_img', action='store_true', default=True,
	                    help='use multi images or single image')

    parser.add_argument('-P', '--imgpath', type=str, 
    default='/home/cyyan/projects/ccRCC/data/test/non/',
	                    help='the url/path of image(s) to generate feature visualization')
	
    parser.add_argument('--imgsize', type = int,
                        default = 512, choices = [200, 224, 256, 512],
                        help = 'the image size, such as 512 or 200 or 256')
    parser.add_argument('-S', '--savepath', type=str, 
    default='/home/cyyan/projects/ccRCC/results/attentionCAM/non/',
	                    help='the overlap images path to save')
    args = parser.parse_args()
    return args


############################main function #####################################
def main():
    finalconv_name = 'layer4'
    # finalconv_name = 'avgpool'
    args =GetArgs()
    
    net = LoadNet(args.imgsize, args.modelpath) ## load the network
    net._modules.get(finalconv_name).register_forward_hook(hook_feature) # get feature maps
    
    if args.multi_img:
        ImgList = os.listdir(args.imgpath)
    else:
        ImgList = [args.imgpath]
        
        
    for num, IMG_URL in enumerate(ImgList):
        if args.multi_img:
             IMG_URL = args.imgpath + IMG_URL
             savename = IMG_URL[len(args.imgpath):-4]
        else:
             savename = args.imgpath.split('/')[-1][0:-4]
             
        if os.path.exists(args.savepath + savename + '.png'):
            continue
        # input image
        torch_img = preprocess(Image.open(IMG_URL)).unsqueeze(0)
        # network prediction
        prob_output = F.softmax(net(torch_img.cuda()), dim = 1)
#        print prob_output
        # argmax of prob
        prediction = np.array(torch.argmax(prob_output, dim=1))[0]


        if prediction == 0:
            weight = GetWeightSoftmax(net, prediction, args.imgsize)
            CAMs = ReturnCAM(features_blobs, weight, args.imgsize, args.imgsize)
            SaveImage(IMG_URL, CAMs, args.savepath, savename)
        else:
            print(IMG_URL, 'prediction wrong')

    
if __name__ == '__main__':
    main()