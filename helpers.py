import torch
from torch import nn

from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# simple mlp network
def model(input, output, width=64):
    network = nn.Sequential(
        nn.Linear(input, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, output),
        nn.Sigmoid()
    )

    return network.to(device)


# initialize pointcloud
def init_points(n_points_init=100, center=torch.zeros(3), features=3):
    xyz = center.reshape([1, 3]).tile([n_points_init, 1])
    rgb = torch.ones([n_points_init, features])

    xyz = nn.Parameter(xyz.to(device), requires_grad=True)
    rgb = nn.Parameter(rgb.to(device), requires_grad=True)

    return xyz, rgb


# scale number of points up by ratio and point size down by factor
def update_point_resolution(xyz, rgb, radius, ratio=10, factor=2):
    # update number of points and radius
    print('update number of points and radius')
    xyz = xyz.tile([ratio, 1])
    rgb = rgb.tile([ratio, 1])
    xyz = nn.Parameter(xyz, requires_grad=True)
    rgb = nn.Parameter(rgb, requires_grad=True)
    radius /= factor

    print('number of points: %d' % len(xyz))
    print('radius: %f' % radius)

    # update point optimizer
    lr = 1e-3 if len(xyz) < 10**5 else 1e-4
    opt = torch.optim.Adam([xyz, rgb], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, verbose=True)

    return xyz, rgb, radius, opt, sched


# save pointcloud reconstruction progress
def save_progress(xyz, rgb, radius, model, savefile, save_ply=False):
    savedict = {'xyz' : xyz.detach().cpu(), 'features' : rgb.detach().cpu(), 'radius' : radius}
    if model: savedict['network'] = model.state_dict()
    torch.save(savedict, savefile)

    if save_ply:
        plyrgb = (rgb[...,:3] * 255).int().detach().cpu()
        savepcl = Pointclouds(points=[xyz.detach().cpu()], features=[plyrgb])
        IO().save_pointcloud(savepcl, savefile.replace('.pth', '.ply'))


# positional encoding
def posenc(x, L=8):
    output = [x] + [ enc(2.0**i * x) for enc in (torch.sin, torch.cos) for i in range(L) ]
    return torch.cat(output, -1)


# psnr metric
def psnr(gt, pred, maxpixel=1.0):
    mse = ((gt - pred) **2).mean(dim=[1,2,3])
    psnr = 20 * torch.log10( maxpixel / torch.sqrt(mse) )
    return float(psnr.mean())