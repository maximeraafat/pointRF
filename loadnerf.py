import os
import json
import torch
import numpy as np

from pytorch3d.renderer import PerspectiveCameras

from visualizer import load_img


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# pytorch3d camera from nerf transforms.json
def transforms_cam(path):
    with open(path, 'r') as filename:
        data = json.load(filename)

    angle = data['camera_angle_x']
    transforms = torch.Tensor([ frame['transform_matrix'] for frame in data['frames'] ]) # .to(device)

    rotations = []
    translations = []
    focals = []

    # code from load_blender_data() function in pytorch3d/implicitron/dataset/load_blender.py
    # and _interpret_blender_cameras() function in pytorch3d/implicitron/dataset/single_sequence_dataset.py
    for pose in transforms:
        f = 1 / np.tan(0.5 * angle)
        f = torch.FloatTensor([f, f])

        pose = pose[:3, :4]
        matrix = torch.eye(4, dtype=pose.dtype)
        matrix[:3, :3] = pose[:3, :3].t()
        matrix[3, :3] = pose[:, 3]
        matrix = matrix.inverse()

        matrix[:, [0, 2]] *= -1 # flip xz coordinates

        R, T = matrix[:, :3].split([3, 1], dim=0)

        rotations.append(R)
        translations.append(T)
        focals.append(f)

    rotations = torch.stack(rotations)
    translations = torch.cat(translations)
    focals = torch.stack(focals)

    cameras = PerspectiveCameras(focal_length=focals, R=rotations, T=translations).to(device)

    del data, angle, transforms, rotations, translations, focals
    return cameras


# images tensor from nerf transforms.json
def transforms_img(path, alpha=True, background=[0, 0, 0]):
    folderpath = os.path.dirname(path)

    with open(path, 'r') as filename:
        data = json.load(filename)

    imgpaths = [frame['file_path'] for frame in data['frames']]
    exts = ['', '.png', '.jpg', '.jpeg']

    for ext in exts:
        try:
            alpha *= ( load_img(os.path.join(folderpath, imgpaths[0]) + ext).shape[-1] == 4 )
            break
        except:
            pass

    images = []
    background = torch.Tensor(background + [0]) # add arbitrary alpha channel

    for imgpath in imgpaths:
        img = load_img( os.path.join(folderpath, imgpath) + ext )
        if not alpha:
            img[...,:3] *= img[...,3:4]
            img += (1.0 - img[...,3:4]) * background
        images.append(img)

    images = torch.stack(images).to(device)

    del data, imgpaths, background, folderpath
    return images if alpha else images[...,:3]