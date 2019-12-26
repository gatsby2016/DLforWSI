import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from glob import glob
import openslide
from openslide import deepzoom
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import io
from _fullyConvResNet34 import *
# import scipy

'''
# get tiles from one slide by deepzoom method
# input: '../data/slides/11-04622  CD.mrxs'
def slide2tile_dz(slidepath, tilesize=512, readlevel=0):
    slide = get_wsi_info(slidepath)
    dzi = deepzoom.DeepZoomGenerator(slide, tile_size=tilesize, overlap=0, limit_bounds=False)
    level = dzi.level_count - readlevel - 1
    tile_num_x, tile_num_y = dzi.level_tiles[level]
    return dzi.level_count, tile_num_x, tile_num_y, dzi
'''


# open the whole slide image and reach information
def get_wsi_info(slidepath):
    slide = openslide.open_slide(slidepath)
    print('Numbers of level in this WSI: ', slide.level_count)
    print('Dimensions of all levels in this WSI (width, height):\n ', slide.level_dimensions)
    return slide


# filter the white background
def filter_background(dpslide, predlevel, downsample=8):
    width, height = dpslide.level_dimensions[predlevel+downsample]
    lowlevelWSI = dpslide.read_region((0, 0), level=predlevel+downsample, size=(width, height))
    lowlevelWSI = np.transpose(np.array(lowlevelWSI)[:, :, 0:3], [1, 0, 2]) # get rgb from rgba and transpose
    lowlevelWSI[lowlevelWSI == 0] = 255 # fill the zero region to 255
    gray = cv2.cvtColor(lowlevelWSI[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY) # rgb2bgr then gray
    value, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)
    # plt.imshow(thresh)
    # plt.show()
    return thresh


# image pre-processing
preprocess = transforms.Compose([transforms.ToTensor(),  # preprocess was operated on original image
                                transforms.Normalize([0.787, 0.5723, 0.769], [0.1193, 0.1878, 0.0974])])


# load the network
def load_net(modelpath, numclasses=2):
    net = resnet34(pretrained=False, num_classes=numclasses)
    net.load_state_dict(torch.load(modelpath))
    net.cuda()
    net.eval()
    # print(net)
    return net


# network and image for prediction
def net_prediction(img, net):
    torch_img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        torch_img = torch_img.cuda()
        predProb = F.softmax(net(torch_img), dim=1)
        # predProb = net(torch_img)
        # predBina = torch.argmax(predProb, dim=1)
    return predProb.squeeze(0).cpu().detach().numpy() #, predBina.cpu().numpy()


# main() by read_region method of openslide
def main(slidepath, modelpath, predlevel=0, traininput=512, testinput=512*10, downfactor=32, numclass=2):
    slide = get_wsi_info(slidepath)

    width, height = slide.level_dimensions[predlevel]

    winstep = testinput - traininput + downfactor # 8224
    testoutput = (testinput - traininput)//downfactor + 1

    vis = np.zeros((numclass, (width//winstep+1)*testoutput, (height//winstep+1)*testoutput))
    print('Visualization results shape: ', vis.shape)

    model = load_net(modelpath)
    for j in range(0, height, winstep):
        for i in range(0, width, winstep):
            image = slide.read_region((i, j), level=predlevel, size=(min(testinput, width - i), min(testinput, height - j)))
            image = np.array(image)[:, :, 0:3]
            image = np.transpose(image, [1, 0, 2])
            # image[image == 0] = 255

            Prob = net_prediction(image, model)
            _, thisw, thish = Prob.shape
            # print('thisw: ', thisw, 'thish: ', thish)

            blocki = i//winstep
            blockj = j//winstep
            print('column:', blocki, 'row: ', blockj)
            vis[:, blocki*testoutput:blocki*testoutput+thisw, blockj*testoutput:blockj*testoutput+thish] = Prob

    vis = vis[:, 0: vis.shape[1]-testoutput+thisw, 0: vis.shape[2]-testoutput+thish] # the last thisw and thish is 153, 185
    print('Visualization results shape: ', vis.shape)

    thresholdimg = filter_background(slide, predlevel)
    newthresh = cv2.resize(thresholdimg, vis.shape[::-1][:2], interpolation=cv2.INTER_NEAREST)

    background = newthresh/2.38 # get the background to 0.42
    newthresh = ~(newthresh*255)//255 # getting the tissue region filtering the background
    vis = vis * newthresh + background

    return np.transpose(vis, [2, 1, 0])


if __name__ == "__main__":
    modelpth = '../modelFullyConv/epoch_7.pkl'
    slidepth = '../data/slides/11-04622  CD.mrxs'
    start = time.time()
    stride = 32
    train_input = 512
    test_inpt = 512 * 17
    visual = main(slidepth, modelpth, predlevel=0, traininput=train_input, testinput=test_inpt, downfactor=stride, numclass=2)

    np.save('../results/fullyConv/vis_FullyConv.npy', visual)
    io.savemat('../results/fullyConv/vis_FullyConv.mat', {'prob': visual})
    print('Finished! Time consuming (sec): ', time.time() - start)
    #
    # visual = np.load('../results/fullyConv/vis_FullyConv.npy')
    plt.imshow(visual[:,:,1])
    plt.show()
