import sklearn.cluster as scluter
from scipy import io
import numpy as np
import h5py
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from glob import glob
import openslide
from openslide import deepzoom
import time
from _fullyConvResNet34 import *
# import scipy


'''
path = '../results/feats.mat'
mat = h5py.File(path)
print(list(mat.keys()))
feats = np.array(mat.get('deepfeats')).transpose()
print(feats.shape)

feats_pca = PCA(n_components=100).fit_transform(feats)
print(feats_pca.shape)


centroids, assignments, _, best_iter = scluter.k_means(feats_pca, 3, verbose=True, return_n_iter=True)
# print('centroids:', centroids)
# print('best_iter: ', best_iter)
print(assignments)


fh = open('/home/cyyan/projects/ccRCC/data/test.txt', 'r')
labels = np.array([])
for line in fh:
    line = line.rstrip()
    words = line.split(' ')[-1]
    labels = np.append(labels, int(words))
print(labels.shape)
fig = plt.figure(figsize=(5,10))
ax = Axes3D(fig, rect=[0,0,1,1], elev=30, azim=20)
plt.scatter(feats_pca[:,0], feats_pca[:,1], feats_pca[:,2], c=assignments, marker='.')
plt.show()
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


# hook the feature extractor
features_blobs = [0]
def hook_feature(module, input, output):
    #    features_blobs.append(output.data.cpu().numpy())
    features_blobs[0] = output.data.cpu().numpy()


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
        # fconv = net.Fconv(torch_img)

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
    featsVis = np.zeros((3, (width//winstep+1)*testoutput, (height//winstep+1)*testoutput))
    print('Visualization results shape: ', vis.shape)

    model = load_net(modelpath)
    model._modules.get('Fconv').register_forward_hook(hook_feature)  # get feature maps

    for j in range(0, height, winstep):
        for i in range(0, width, winstep):
            image = slide.read_region((i, j), level=predlevel, size=(min(testinput, width - i), min(testinput, height - j)))
            image = np.array(image)[:, :, 0:3]
            image = np.transpose(image, [1, 0, 2])
            # image[image == 0] = 255

            Prob = net_prediction(image, model)
            features = features_blobs[0].squeeze()
            # newfeats = np.reshape(features, (features.shape[0], features.shape[1]*features.shape[2]))
            # newfeats = newfeats.transpose()
            # feats_pca = PCA(n_components=3).fit_transform(newfeats)
            # feats_pca = feats_pca.transpose()
            # feats_pca = np.reshape(feats_pca, (feats_pca.shape[0],features.shape[1], features.shape[2]))
            # print(feats_pca.shape)

            # centroids, assignments, _, best_iter = scluter.k_means(feats_pca, 3, verbose=True, return_n_iter=True)
            #

            _, thisw, thish = Prob.shape
            # print('thisw: ', thisw, 'thish: ', thish)

            blocki = i//winstep
            blockj = j//winstep
            print('column:', blocki, 'row: ', blockj)
            vis[:, blocki*testoutput:blocki*testoutput+thisw, blockj*testoutput:blockj*testoutput+thish] = Prob
            featsVis[:, blocki*testoutput:blocki*testoutput+thisw, blockj*testoutput:blockj*testoutput+thish] = features


    vis = vis[:, 0: vis.shape[1]-testoutput+thisw, 0: vis.shape[2]-testoutput+thish] # the last thisw and thish is 153, 185
    featsVis = featsVis[:, 0: featsVis.shape[1]-testoutput+thisw, 0: featsVis.shape[2]-testoutput+thish] # the last thisw and thish is 153, 185

    print('Visualization results shape: ', vis.shape)
    print('feats results shape: ', featsVis.shape)

    # thresholdimg = filter_background(slide, predlevel)
    # newthresh = cv2.resize(thresholdimg, vis.shape[::-1][:2], interpolation=cv2.INTER_NEAREST)
    #
    # background = newthresh/2.38 # get the background to 0.42
    # newthresh = ~(newthresh*255)//255 # getting the tissue region filtering the background
#    vis = vis * newthresh + background

    return np.transpose(vis, [2, 1, 0]), np.transpose(featsVis, [2, 1, 0])


if __name__ == "__main__":
    modelpth = '../modelFullyConv/epoch_7.pkl'
    slidepth = '../data/slides/11-04622  CD.mrxs'
    start = time.time()
    stride = 32
    train_input = 512
    test_inpt = 512 * 17
    res, featsVis = main(slidepth, modelpth, predlevel=0, traininput=train_input,
                  testinput=test_inpt, downfactor=stride, numclass=2)

    np.savez('../results/fullyConv/vis_PCAfeats.npz', res, featsVis)
    io.savemat('../results/fullyConv/vis_PCAfeats.mat', {'prob': res, 'featsVis': featsVis})
    print('Finished! Time consuming (sec): ', time.time() - start)

    # data = np.load('../results/fullyConv/vis_PCAfeats.npz')
    # res = data['arr_0']
    # featsVis = data['arr_1']

    feats = np.reshape(featsVis, (featsVis.shape[0] * featsVis.shape[1], featsVis.shape[2]))
    feats = PCA(n_components=3).fit_transform(feats)
    res = np.argmax(np.reshape(res, (res.shape[0] * res.shape[1], res.shape[2])), axis=1)

    # randoms = np.random.random_integers(0, feats.shape[0], feats.shape[0]//100)
    randoms = range(0, feats.shape[0], 10000)
    # fig = plt.figure(figsize=(5, 10))
    # ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(feats[randoms, 0], feats[randoms, 1], c=res[randoms], marker='.')
    # plt.scatter(feats[randoms, 0], feats[randoms, 1], feats[randoms, 2], c=res[randoms], marker='.')
    plt.show()

    # plt.imshow()
    # plt.show()
