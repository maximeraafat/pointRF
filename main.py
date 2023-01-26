import glob
import torch
import argparse

from train import pointoptim
from test import testrenders
from loadnerf import *
from helpers import *


parser = argparse.ArgumentParser()

# training and testing options
parser.add_argument('--datapath', metavar='PATH', default=None, type=str, help='path to the directory containing training, testing and validation transforms.json and images')
parser.add_argument('--train', action='store_true', help='whether to train the point model')
parser.add_argument('--test', action='store_true', help='whether to create testing images and evaluate psnr against the ground truth')
parser.add_argument('--testgif', action='store_true', help='whether to create a gif from the testing images')
parser.add_argument('--testoutput', metavar='PATH', default=None, type=str, help='path to the directory in which the testing images will be stored')
parser.add_argument('--testmdodel', metavar='PATH', default=None, type=str, help='path to a .pth model checkpoint file to be used for testing')

# saving options
parser.add_argument('--savepath', metavar='PATH', default=None, type=str, help='path to the directory in which the model checkpoints will be stored')
parser.add_argument('--saveply', action='store_true', help='whether to save the pointclouds as .ply files')

# training parameters
parser.add_argument('--epochs', metavar='INT', default=30000, type=int, help='number of training epochs')
parser.add_argument('--valfreq', metavar='INT', default=1000, type=int, help='validation frequency during training')
parser.add_argument('--savefreq', metavar='INT', default=1000, type=int, help='save frequency during training')
parser.add_argument('--background', default='[0, 0, 0]', type=str, help='background color')
parser.add_argument('--imagesize', metavar='INT', default=None, type=int, help='training images size')
parser.add_argument('--npoints', metavar='INT', default=100, type=int, help='initial number of points')
parser.add_argument('--initlr', metavar='FLOAT', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--radius', metavar='FLOAT', default=0.1, type=float, help='initial point radius')
parser.add_argument('--finalradius', metavar='FLOAT', default=None, type=float, help='final point radius') # 0.005
parser.add_argument('--radiance', action='store_true', help='whether to train an MLP to learn view dependent colors for each point')

args = parser.parse_args()


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    background = eval(args.background)

    # assertions
    if not args.datapath or not os.path.isdir(args.datapath): raise Warning('--datapath must be a valid directory')
    if not len(background) == 3: raise Warning('--background list must contain 3 rgb values exactly')
    if not args.npoints <= 10**4: raise Warning('coarse to fine optimisation requires a maximum of 1000 points as initialisation')
    if not args.initlr >= 1e-5:  raise Warning('initial learning rate must be bigger or equal than 1e-5')
    if args.saveply and args.radiance: raise Warning('--saveply option is only available for rgb pointclouds')

    dataname = os.path.normpath(args.datapath).split('/')[-1]

    savepath = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'save') if args.savepath is None else args.savepath
    savepath = os.path.join(savepath, dataname)

    if args.train:
        print('\nloading training data...')
        trainpath = os.path.join(args.datapath, 'transforms_train.json')
        if not os.path.isfile(trainpath): raise Warning('%s file not found' % trainpath)
        trainCam = transforms_cam(trainpath)
        trainImg = transforms_img(trainpath, alpha=True, background=background)

        valpath = os.path.join(args.datapath, 'transforms_val.json')
        validation = os.path.isfile(valpath)
        valCam = transforms_cam(valpath) if validation else None
        valImg = transforms_img(valpath, alpha=True, background=background) if validation else None

        alpha = trainImg.shape[-1] == 4
        nfeatures = 32 if args.radiance else 3 + alpha
        xyz, rgb = init_points(n_points_init=args.npoints, features=nfeatures)
        pointoptim(xyz, rgb, trainCam, trainImg, image_size=args.imagesize, epochs=args.epochs, val_freq=args.valfreq, radius=args.radius, final_radius=args.finalradius, background=background, save_freq=args.savefreq, init_lr=args.initlr, save_path=savepath, save_ply=args.saveply, val_cameras=valCam, val_images=valImg)

        del trainCam, trainImg, valCam, valImg

    if args.test:
        print('\nloading testing data...')
        testpath = os.path.join(args.datapath, 'transforms_test.json')
        if not os.path.isfile(testpath): raise Warning('%s file not found' % testpath)
        testCam = transforms_cam(testpath)
        testImg = transforms_img(testpath, alpha=False, background=background)

        testoutput = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'test') if args.testoutput is None else args.testoutput
        testoutput = os.path.join(testoutput, dataname)

        print('test data will be saved under %s' % testoutput)
        if args.testmdodel:
            model = torch.load(args.testmdodel, map_location=device) # load provided test model
        else:
            modelpaths = glob.glob(os.path.join(savepath, '*.pth'))
            if len(modelpaths) == 0: raise Warning('no model checkpoint found in %s' % savepath)
            model = torch.load(max(modelpaths, key=os.path.getctime), map_location=device) # load last saved model

        output = testrenders(model, testCam, image_size=testImg.shape[-2], background=background, savepath=testoutput, savegif=args.testgif)
        print('test psnr = %f' % psnr(testImg, output))

    return


if __name__ == '__main__':
    main()