2019/11/2 update
### projects
ccRCC tumor detection by sliding widows on whole slides images

### data
In total of 5 slides [4 for training and 1 one for validation]
samples distribution
training: [non]23824 + [tumor]40946 
validate: [non]10136 + [tumor]12605
with samples selection manually slightly. Mainly for whole white patches
no stain normalizaiton, no samples agumentation

### methods
DL [resnet34 + weighted cross entory loss] for 45 epochs with 0.001 lr on pretrained ImageNet
weights are: [0.7, 0.3]

### results on epoch 45
 Matrix |  predict
        |  non      tumor
--------|--------------
real non|  7451     2685        =  10136
   tumor|  11     12594        =  12605
-----------------------
ACC: 0.8814, take about 302 seconds



acc = 7451+ 12594 / (10136+12605)= 0.88144761
sen = recall = 12594/ 12605 = 0.999127
spc = 7451 / 10136 = 0.7351
precision = 12594 / (12594+2685) = 0.8242
