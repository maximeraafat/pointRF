import os
import time
import torch

import numpy as np
from torchvision.transforms import Resize

from pytorch3d.structures import Pointclouds

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from visualizer import visualize
from renderer import point_renderer
from helpers import *


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# point color and geometry optimization
def pointoptim(xyz, rgb, cameras, images, image_size, epochs, val_freq, radius, final_radius, background, save_freq, init_lr, save_path, save_ply, val_cameras, val_images):

    totalloss = 0.0
    l1loss = torch.nn.L1Loss()
    os.makedirs(save_path, exist_ok=True)
    validation = not (val_cameras is None or val_images is None)

    fullres = False
    default_size = tuple(images.shape[1:3])
    image_size = default_size if image_size is None else image_size

    # resize images
    if image_size != default_size: fullres_images = images.clone()
    images = Resize(image_size)(images.permute(0, 3, 1, 2)) # shape to (b, c, w, h)
    images = images.permute(0, 2, 3, 1) # shape back to (b, w, h, c)

    # initialize renderer
    background += (images.shape[-1] == 4) * [0]
    renderer = point_renderer(cameras[0], image_size, radius=radius, background=background)
    if validation: vrenderer = point_renderer(cameras[0], default_size, radius=radius, background=background)

    # geometry and color optimizer
    opt = torch.optim.Adam([xyz, rgb], lr=init_lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, verbose=True)

    # view dependent color network optimizer
    colormodel = None
    if rgb.shape[-1] > 4:
        encoding_length = 8
        colormodel = model(rgb.shape[-1] + 3 + 2 * 3 * encoding_length, len(background))
        modelopt = torch.optim.Adam(colormodel.parameters(), lr=1e-3)
        modelsched = torch.optim.lr_scheduler.ReduceLROnPlateau(modelopt, patience=5, verbose=False)
        colormodel.train()

    start = time.time()

    # training loop
    for i in tqdm(range(epochs)):
        idx = np.random.choice(len(images))
        ygt = images[idx]
        cam = cameras[idx]

        if rgb.shape[-1] > 4:
            # view dependent colors
            encoded_xyz = cam.transform_points(xyz)
            encoded_xyz = posenc(encoded_xyz, encoding_length)
            col = colormodel(torch.cat([encoded_xyz, rgb], dim=1))
            pcl = Pointclouds(points=[xyz], features=[col])
        else:
            # diffuse colors
            pcl = Pointclouds(points=[xyz], features=[rgb])

        # rendering
        ypred = renderer(pcl, cameras=cam)[0].clamp(0, 1)

        loss = l1loss(ypred, ygt)
        totalloss += float(loss)

        opt.zero_grad()
        if rgb.shape[-1] > 4: modelopt.zero_grad()
        loss.backward()
        opt.step()
        if rgb.shape[-1] > 4: modelopt.step()

        # validation and visualisation
        if i % val_freq == 0:
            if not validation: visualize((ygt, ypred))
            else:
                vsize = len(val_images)
                vidx = np.random.choice(vsize)
                vgt = val_images[vidx]
                vcam = val_cameras[vidx]

                if rgb.shape[-1] > 4:
                    # view dependent color prediction
                    encoded_vxyz = vcam.transform_points(xyz)
                    encoded_vxyz = posenc(encoded_vxyz, encoding_length)
                    vcol = colormodel(torch.cat([encoded_vxyz, rgb], dim=1))
                    vpcl = Pointclouds(points=[xyz], features=[vcol])
                    vpred = vrenderer(vpcl, cameras=vcam)[0].clamp(0, 1)
                else:
                    # diffuse color prediction
                    vpred = vrenderer(pcl, cameras=vcam)[0].clamp(0, 1)

                print('validation psnr = %f' % psnr(vgt[None], vpred[None]))
                visualize((vgt, vpred))

        # saving
        savefile = os.path.join(save_path, 'epoch_%05d.pth' % i)
        if i % save_freq == 0 and len(xyz) >= 10**6 or i == epochs - 1:
            save_progress(xyz, rgb, radius, colormodel, savefile, save_ply)

        # scheduling
        if i % 100 == 0 and i > 0:
            avgloss = totalloss / 100
            sched.step(avgloss)
            if rgb.shape[-1] > 4: modelsched.step(avgloss)
            totalloss = 0
            end = time.time() - start

            print('loss, time, epoch = (%f, %f, %d)' % (avgloss, end, i)) # print progress so far

            # update model
            if opt.param_groups[0]['lr'] <= 1e-5 and len(xyz) < 10**6:
                save_progress(xyz, rgb, radius, colormodel, savefile, save_ply) # saving
                xyz, rgb, radius, opt, sched = update_point_resolution(xyz, rgb, radius, 10, 2) # 10x points and 1/2 radius
                renderer = point_renderer(cam, image_size, radius=radius, background=background)
                if validation: vrenderer = point_renderer(cam, default_size, radius=radius, background=background)

                # reset color network optimizer
                if rgb.shape[-1] > 4:
                    modelopt = torch.optim.Adam(colormodel.parameters(), lr=1e-3)
                    modelsched = torch.optim.lr_scheduler.ReduceLROnPlateau(modelopt, patience=5, verbose=False)

            # last radius update and full resolution
            if opt.param_groups[0]['lr'] <= 1e-5 and len(xyz) >= 10**6 and not fullres:
                fullres = True
                save_progress(xyz, rgb, radius, colormodel, savefile, save_ply) # saving

                if final_radius:
                    radius = final_radius
                    print('radius: %f' % radius)

                if image_size != default_size:
                    images = fullres_images
                    image_size = default_size
                    print('train at full resolution:', image_size)

                renderer = point_renderer(cam, image_size, radius=radius, background=background)
                if validation: vrenderer = point_renderer(cam, default_size, radius=radius, background=background)

                opt = torch.optim.Adam([xyz, rgb], lr=1e-4)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, verbose=True)