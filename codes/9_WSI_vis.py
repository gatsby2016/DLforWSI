import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from glob import glob
import openslide
from openslide import deepzoom
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import io

# open the whole slide image and reach information
def get_wsi_info(slidepath):
    slide = openslide.open_slide(slidepath)
    print('Numbers of level in this WSI: ', slide.level_count)
    print('Dimensions of all levels in this WSI (width, height):\n ', slide.level_dimensions)
    return slide


# get tiles from one slide by deepzoom method
# input: '../data/slides/11-04622  CD.mrxs'
def slide2tile_dz(slidepath, tilesize=512, readlevel=0):
    slide = get_wsi_info(slidepath)
    dzi = deepzoom.DeepZoomGenerator(slide, tile_size=tilesize, overlap=0, limit_bounds=False)
    level = dzi.level_count - readlevel - 1
    tile_num_x, tile_num_y = dzi.level_tiles[level]
    return dzi.level_count, tile_num_x, tile_num_y, dzi


# image pre-processing
preprocess = transforms.Compose([transforms.ToTensor(),  # preprocess was operated on original image
                                transforms.Normalize([0.787, 0.5723, 0.769], [0.1193, 0.1878, 0.0974])])


# load the network
def load_net(modelpath, numclasses=2):
    net = models.resnet34(pretrained=False, num_classes=numclasses)
    net.load_state_dict(torch.load(modelpath))
    net.cuda()
    return net.eval()


# network and image for prediction
def net_prediction(img, net):
    torch_img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        torch_img = torch_img.cuda()
        predProb = F.softmax(net(torch_img), dim=1)
        predBina = torch.argmax(predProb, dim=1)
    return predProb.squeeze().cpu().detach().numpy(), predBina.cpu().numpy()


# main() by deepzoom method of openslide
def main_dz(slidepath, modelpath, predlevel=0, tilesize=512, numclass=2):
    level, num_x, num_y, sample = slide2tile_dz(slidepath, tilesize=tilesize, readlevel=predlevel)

    vis = np.empty((num_x, num_y, numclass))
    print('Visualization results shape: ', vis.shape)
    for i in range(num_x):
        for j in range(num_y):
            image = np.array(sample.get_tile(level-1, (i, j)))

            # here the tumor region about 12 and non > 12, so if half of one image are tissue is ok.
            threshold = np.mean(np.std(image, axis=2))
            # print(threshold)
            if threshold < 6:
                continue

            Prob, Bina = net_prediction(image, load_net(modelpath))
            print('row:', i, 'column: ', j, 'pred: ', Bina)
            vis[i, j, :] = Prob

    return vis


# main() by read_region method of openslide
def main_rr(slidepath, modelpath, predlevel=0, windowstep=100, tilesize=512, numclass=2):
    slide = get_wsi_info(slidepath)
    width, height = slide.level_dimensions[predlevel]

    vis = np.empty(((width-tilesize)//windowstep+1, (height-tilesize)//windowstep+1, numclass))
    print('Visualization results shape: ', vis.shape)
    for i in range(0, width-tilesize, windowstep):
        for j in range(0, height-tilesize, windowstep):
            rgba = slide.read_region((i, j), level=predlevel, size=(tilesize, tilesize))
            image = np.array(rgba)[:, :, 0:3]
            # plt.imshow(image)
            # plt.show()

            # here the tumor region about 12 and non > 12, so if half of one image are tissue is ok.
            threshold = np.mean(np.std(image, axis=2))
            # print(threshold)
            if threshold < 6:
                continue

            Prob, Bina = net_prediction(image, load_net(modelpath))
            print('column:', i, 'row: ', j, 'pred: ', Bina)
            vis[i//windowstep, j//windowstep, :] = Prob
    return vis

if __name__ == "__main__":
    modelpth = '../model/epoch_45.pkl'
    slidepth = '../data/slides/11-04622  CD.mrxs'
    start = time.time()
    visual = main_dz(slidepth, modelpth, predlevel=0, tilesize=512, numclass=2)
    # visual = main_rr(slidepth, modelpth, predlevel=0, windowstep=100, tilesize=512, numclass=2)
    #
    np.save('../results/vis.npy', visual)
    io.savemat('../results/vis.mat', {'prob': visual})
    print('Finished! Time consuming (sec): ', time.time() - start)
    #
    # visual = np.load('../results/vis100.npy')
    # plt.imshow(visual[:,:,1])
    # plt.show()

