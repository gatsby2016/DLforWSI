# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:15:24 2018
@author: yann
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ### cal roc and auc
# from sklearn import cross_validation
import time
import math
import os


### load the testdata and the network
### input is datapath, modelpath, batch_size and image size
### output is the data laoder and model net
def LoadDataAndNet(DataPath, batch_size, ModelPath, device):
    ImgSize = 512
    ######################################### data loader
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        #            transforms.Normalize([0.4923, 0.3272, 0.4164], [0.3735, 0.2936, 0.3243])
        #                    transforms.Normalize([0.3256, 0.2218, 0.2782], [0.3750, 0.2831, 0.3252])
        #                    transforms.Normalize([0.1809, 0.1263, 0.1561], [0.3070, 0.2318, 0.2680])
        transforms.Normalize([0.0612, 0.0423, 0.0525], [0.1900, 0.1396, 0.1646])
    ])
    testset = torchvision.datasets.ImageFolder(root=DataPath, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    ######################################### define and load network
    net = torchvision.models.resnet50(num_classes=2)
    net.load_state_dict(torch.load(ModelPath))
    net.to(device)
    print('Data and Net have been loaded.........   Using: ', device)
    return testloader, net


### define the testdata function by deep network, you can assign the net
### input is data testloader and the model net
### output is all data real label, prediction assignment and score
def TestDataByDeepNet(testloader, net, device):
    prediction = np.array([])
    real = np.array([])
    score = np.array([])
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            print(i)
            images, labels = data
            #            print labels
            images, labels = images.to(device), labels.to(device)
            #            outputs = net(images)
            #            _, predicted = torch.max(outputs.data, 1)
            #            score      = np.concatenate((score,      outputs.data.cpu().numpy()[:,1]),axis = 0)
            #            prediction = np.concatenate((prediction, np.array(predicted)),axis = 0)
            #            real       = np.concatenate((real,       np.array(labels)),axis = 0)
            #            total += labels.size(0)
            #            correct += (predicted == labels).sum().item()
            outputs = F.softmax(net(images), dim=1)
            #            print outputs #.data.cpu()
            pred_ind = torch.argmax(outputs, 1)
            #            print pred_ind
            score = np.concatenate((score, np.array(outputs[:, 1])), axis=0)
            prediction = np.concatenate((prediction, np.array(pred_ind)), axis=0)
            real = np.concatenate((real, np.array(labels)), axis=0)
        accuracy = sum(prediction == real) * 1.0 / len(real)
        print('Test Accuracy is: ', accuracy)
    return score, prediction, real, accuracy


### calculate and show the confusion matrix of data
### input is the real label and predicition assignment
### output is the print result
def ShowConfusionMatrix(real, prediction):
    T0 = ((real == 0) & (prediction == 0)).sum()
    F1 = ((real == 0) & (prediction == 1)).sum()
    T1 = ((real == 1) & (prediction == 1)).sum()
    F0 = ((real == 1) & (prediction == 0)).sum()
    print('-----------------------')
    print('        |  predict   ')
    print('        |  non      tumor')
    print('--------|--------------')
    print('real non| ', T0, '   ', F1, '       = ', T0 + F1)
    print('   tumor| ', F0, '   ', T1, '       = ', T1 + F0)
    print('-----------------------')


### AUC value and roc curve
### input is real label and predicted score, the result is a pobability
### output the auc value and roc curve plot
def ROCAUC(real, score):
    fpr, tpr, _ = roc_curve(real, score, pos_label=1)  ###计算真正率和假正率
    AUC = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='red', lw=2, label='AUC = %0.4f' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return AUC


if __name__ == "__main__":
    ######################################### input parameters
    batch_size = 100
    DataPath = '/media/yann/FILE/Kidney/Yan/Ablation_Study/ResNet40_Fourth/data/test/'
    ModelPath = '/media/yann/FILE/Kidney/Yan/Ablation_Study/ResNet40_Fourth/model/resnet_params_5.pkl'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(time.ctime())
    ######################################### perform the network prediction
    testloader, net = LoadDataAndNet(DataPath, batch_size, ModelPath, device)
    score, prediction, real, accuracy = TestDataByDeepNet(testloader, net, device)
    np.savez('../result/test_result_40xfourth_0508.npz', score=score, prediction=prediction, real=real)
    ShowConfusionMatrix(real, prediction)
    AUC = ROCAUC(real, score)
    print(time.ctime())