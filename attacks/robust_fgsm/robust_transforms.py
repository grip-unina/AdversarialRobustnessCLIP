'''
from: https://github.com/paarthneekhara/AdversarialDeepFakes
'''

import torch
import torch.nn as nn
from torch import autograd
import numpy
from torchvision import transforms
import torchgeometry as tgm
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective
    

def compress_decompress(image,factor,cuda=True):
  # :param image: tensor image
  # :param cuda: enables cuda, must be the same parameter as the model
  # using interpolate(input, size, scale_factor, mode, align_corners) to upsample
  # uses either size or scale factor

  image_size = list(image.size())
  image_size = image_size[2:]
  compressed_image = nn.functional.interpolate(image, scale_factor = factor, mode = "bilinear", align_corners = True)
  decompressed_image = nn.functional.interpolate(compressed_image, size = image_size, mode = "bilinear", align_corners = True)

  return decompressed_image



def add_gaussian_noise(image,std,cuda=True):
  # :param image: tensor image
  # :param amount: amount of noise to be added
  # :param cuda: enables cuda, must be the same parameter as the model


  new_image = image + std*torch.randn_like(image)
  new_image = torch.clamp(new_image, min=0, max=1)

  return new_image


def gaussian_blur(image,kernel_size=(11,11),sigma=(10.5, 10.5),cuda=True):
  # smooths the given tensor with a gaussian kernel by convolving it to each channel. 
  # It suports batched operation.
  # :param image: tensor image
  # :param kernel_size (Tuple[int, int]): the size of the kernel
  # :param sigma (Tuple[float, float]): the standard deviation of the kernel

  gauss = tgm.image.GaussianBlur(kernel_size, sigma)

  # blur the image
  img_blur = gauss(image)

  # convert back to numpy. Turned off because I'm returning tensor
  #image_blur = tgm.tensor_to_image(img_blur.byte())

  return img_blur


def translate_image(image, shift_x = 20, shift_y = 20, cuda=True):

  image_size = list(image.size())
  image_size = image_size[2:]
  


  
  h, w = image_size[0], image_size[1]  


  #because height indexed from zero
  points_src = torch.FloatTensor([[
    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
  ]])

  # the destination points are the image vertexes
  points_dst = torch.FloatTensor([[
    [0 + shift_x, 0 + shift_y], [w-1+shift_x, 0 + shift_y], [w-1+shift_x, h-1 + shift_y], [0 + shift_x, h-1 + shift_y],]])

  if cuda:
    points_src = points_src.cuda()
    points_dst = points_dst.cuda()
  # compute perspective transform
  M = get_perspective_transform(points_src, points_dst)

  # warp the original image by the found transform
  if cuda:
    image = image.cuda()
    
  img_warp = warp_perspective(image, M, dsize=(h, w))

  return img_warp

















