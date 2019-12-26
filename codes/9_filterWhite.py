import skimage.measure
from skimage.transform import rescale, rotate
from scipy.stats import mode
import numpy as np
from glob import glob
from PIL import Image
# from scipy.misc import imsave
#
# Image.MAX_IMAGE_PIXELS = 1e10


imgurl = '../data/slides/'
imglist = glob(imgurl+'*png')
# URL = imglist[1]
for URL in imglist:
    print(URL)
    img = Image.open(URL)
    img = np.array(img)
    print(np.mean(np.std(img, axis=2)))

## here we can measure the std of 3 channel of each point in one image.
## then, we can average the std value of all points in one image
## the value can be measured for filtering the background white
