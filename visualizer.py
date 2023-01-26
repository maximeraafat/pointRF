import os
import torch
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    __IPYTHON__
    ipython = True
except NameError:
    ipython = False


# load image as tensor
def load_img(path):
    return torch.FloatTensor(np.array(Image.open(path))) / 255.0


# save a tensor batch of images
def save_img(tensorbatch, savepath, name='', savegif=False):
    os.makedirs(savepath, exist_ok=True)

    n_images = len(tensorbatch)
    digits = int( np.floor(np.log10(n_images)) + 2 )

    tensorbatch = (tensorbatch * 255).detach().cpu().numpy().astype(np.uint8)

    for i in range(n_images):
        filename = name + '%s.png' % str(i).zfill(digits)
        filepath = os.path.join(savepath, filename)
        img = Image.fromarray(tensorbatch[i])
        img.save(filepath)

    if savegif:
        gifname = os.path.basename( os.path.normpath(savepath) )
        gifpath = os.path.join(savepath, '%s.gif' % gifname)
        imageio.mimsave(gifpath, tensorbatch)


# visualize a tensor or a sequence of tensors
def visualize(images, size=(8,8), savefile=None):
    if type(images) is not torch.Tensor: images = torch.hstack(images)
    images = images.detach().cpu()
    plt.figure(plt.gcf(), figsize=size)
    plt.imshow(images)
    plt.axis('off')
    plt.show() if ipython else plt.pause(1)
    plt.clf()

    saveimg = (images * 255).numpy().astype(np.uint8)
    if savefile: plt.imsave(savefile, saveimg)