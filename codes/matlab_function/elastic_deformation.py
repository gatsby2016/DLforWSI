#! coding=utf-8
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import os

img2dir = '/home/yann/Projects/G3G4/'#输出图像
mask2dir = '/home/yann/Projects/G3G4/data/'
img_path = '/home/yann/Projects/G3G4'#输入图像
imgmask_path = '/home/yann/Projects/G3G4/data'
ends = '.png'#后缀
N = 3 #生成数量
channel = 3 #mask的通道（1是二值图，3是rgb图）
mask_temp = list()
def main():
    filelist = get_imlist(img_path)
    for img_name in filelist:
        print(filelist.index(img_name), img_name)
        #mask_name = img_name.split('.')[0]
        # mask_name = img_name.split('_')
        # mask_name += 'mask.tif'####
        # [mask_temp.append(name) for name in img_name.split('_')]
        mask_name = img_name
        img = plt.imread(img_path + '/' + img_name)
        mask = plt.imread(imgmask_path + '/' + mask_name)

        for i in range(N):
            per_aug(img, mask, img_name, mask_name, i, channel)
        print('success')

def per_aug(img, mask, img_name, mask_name, add, channel):
    """
    :param img: raw img
    :param mask: groundtruth
    :param img_name: the name of the raw img
    :param mask_name: the name of the groundtruth
    :param add: increased name of output file
    :param channel: the number of groundtruth's channel(1:binary;3:rgb)
    :return: none
    """
# Apply transformation on image
    if channel == 3:
        img, mask = elastic_transform_3(img, mask, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)
    elif channel == 1:
        img, mask = elastic_transform_1(img, mask, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)

# Split image and mask
    (r, g, b) = cv2.split(img)
    img = cv2.merge([b, g, r])
    if os.path.isdir(img2dir):
        pass
    else:
        os.mkdir(img2dir)
    if os.path.isdir(mask2dir):
        pass
    else:
        os.mkdir(mask2dir)
    cv2.imwrite(img2dir + str(add) + '_' + img_name, img)
    cv2.imwrite(mask2dir + str(add) + '_' + mask_name, mask)
# Function to distort image
def elastic_transform_3(image, mask, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],#raw point
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)#output point
    M = cv2.getAffineTransform(pts1, pts2)# affine matrix
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)

def elastic_transform_1(image, mask, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    image = np.concatenate((image, mask[..., None]), axis=2)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)#
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return image[..., :3], image[..., 3:]
def get_imlist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """
    return [f for f in os.listdir(path) if f.endswith(ends)]##

if __name__ == "__main__":
    main()
