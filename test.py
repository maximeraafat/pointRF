import os
import torch
import imageio
import numpy as np
from PIL import Image

from pytorch3d.structures import Pointclouds

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from renderer import point_renderer
from helpers import model as network
from helpers import *


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get test renders
def testrenders(checkpoint, cameras, image_size, background, savepath, savegif=False):
    xyz, rgb, radius = ( checkpoint['xyz'], checkpoint['features'], checkpoint['radius'] )
    # rgb = rgb[..., :3] # when radiance, enable to visualize 3 first point deep features

    if rgb.shape[-1] > 4:
        # view dependent color network
        encoding_length = 8
        channels =  list(checkpoint['network'].values())[-1].shape[0] # output dimension of last weight or bias tensor in network
        colormodel = network(rgb.shape[-1] + 3 + 2 * 3 * encoding_length, channels)
        colormodel.load_state_dict(checkpoint['network'])
        colormodel.eval()
    else:
        # diffuse color pointcloud
        pcl = Pointclouds(points=[xyz], features=[rgb[..., :3]]).to(device)

    os.makedirs(savepath, exist_ok=True)

    n_images = len(cameras)
    digits = int( np.floor(np.log10(n_images)) + 2 )

    renderer = point_renderer(cameras[0], image_size, radius=radius, background=background)

    for i in tqdm(range(n_images)):
        cam = cameras[i]

        if rgb.shape[-1] > 4:
            encoded_xyz = cam.transform_points(xyz)
            encoded_xyz = posenc(encoded_xyz, encoding_length)
            col = colormodel(torch.cat([encoded_xyz, rgb], dim=1))
            pcl = Pointclouds(points=[xyz], features=[col[..., :3]])

        img = renderer(pcl, cameras=cam).clamp(0, 1)[0] * 255
        img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))
        filename = '%s.png' % str(i).zfill(digits)
        filepath = os.path.join(savepath, filename)
        img.save(filepath)

    if rgb.shape[-1] > 4: del encoding_length, encoded_xyz, colormodel
    del pcl, xyz, rgb, radius, checkpoint

    renders = []
    for i in range(n_images):
        filename = '%s.png' % str(i).zfill(digits)
        filepath = os.path.join(savepath, filename)
        renders.append( np.array(Image.open(filepath)) )

    if savegif:
        gifname = os.path.basename( os.path.normpath(savepath) )
        gifpath = os.path.join(savepath, '%s.gif' % gifname)
        imageio.mimsave(gifpath, renders)

    return torch.FloatTensor(np.array(renders)).to(device) / 255.0