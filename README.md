## projects / aim
**ccRCC tumor detection by DL sliding widows on whole slides images.**

## data: slides and patches
In total of 5 slides *4 for training and 1 one for validation*

| patches distribution | non   | tumor |
| ---------------------| ----- | ----- |
| training             | 23824 | 40946 |
| validation           | 10136 | 12605 |

The patch size is 512*512 at 20x magnificaiton.
> with samples selection manually slightly. Mainly for whole white patches. 
> No stain normalizaiton, no samples agumentation

## 2019/11/2 update
### methods
- DL `resnet34 + weighted cross entory loss` for `45` epochs with lr `0.001` on **pretrained ImageNet**.
- weights are: [0.7, 0.3]

### results on epoch 45
>ACC: 0.8814, take about 302 seconds

| Confusion Matrix     | non   | tumor | total |
| ---------------------| ----- | ----- | ----- |
| real non             | 7451  | 2685  | 10136 |
| real tumor           | 11    | 12594 | 12605 |

-----------------------


- acc = 7451+ 12594 / (10136+12605)= 0.88144761
- sen = recall = 12594/ 12605 = 0.999127
- spc = 7451 / 10136 = 0.7351
- precision = 12594 / (12594+2685) = 0.8242

## 2019/12/18
### methods 
Modified DL framework with fully convs for fast WSI prediction. 
We replaced **the last GAP and fc.** in **resnet34** with **AP with kernel size 16*16 followed by fconv and classifer_conv with 1*1 kernel **. 

A simple implementation is shown as followed:
```python
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=16, stride=1)
        self.Fconv = nn.Conv2d(512 * block.expansion, 512, kernel_size=1, stride=1, bias=False)
        self.Fbn = norm_layer(512)
        self.final = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False)
```
Training predurce is the same as the previous, but WSI prediction is faster than it.

[reference1:ScanNet: A Fast and Dense Scanning Framework for Metastastic Breast Cancer Detection from Whole-Slide Image](https://ieeexplore.ieee.org/document/8354169)
[reference2:RMDL: Recalibrated Multi-instance Deep Learning for Whole Slide Gastric Image Classification](https://www.researchgate.net/publication/335511979_RMDL_Recalibrated_Multi-instance_Deep_Learning_for_Whole_Slide_Gastric_Image_Classification)

### Visualization results 
one visualization result is the WSI prediction. We compared with the previous.
![WSIprediction](https://github.com/gatsby2016/DLforWSI/blob/master/results/results1.png)
Another visualization result is the PCA and TSNE features points cluster visualization.
![PCAandTSNE](https://github.com/gatsby2016/DLforWSI/blob/master/results/pca.png)


