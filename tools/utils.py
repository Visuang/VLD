import os
import time
import math
from torchvision import transforms

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))



MODALITY = {'RGB/IR': -1, 'RGB': 0, 'IR': 1}
MODALITY_ = {-1:'All', 0: 'RGB', 1: 'IR'}
CAMERA = {'LS3': 0, 'G25': 1, 'CQ1': 2, 'W4': 3, 'TSG1': 4, 'TSG2': 5}


def get_auxiliary_alpha(curr_epoch, max_epoch, phi):
    # return phi
    # return 0.5 * math.exp(-phi * curr_epoch / max_epoch)
    return (math.cos(math.pi * curr_epoch / max_epoch) + phi) / (2 + 2 * phi)




def get_transform(opt, mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(opt.img_hw),
            transforms.Pad(opt.padding),
            transforms.RandomCrop(opt.img_hw),
            transforms.ToTensor(),
            transforms.Normalize(opt.norm_mean, opt.norm_std)
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize(opt.img_hw),
            transforms.ToTensor(),
            transforms.Normalize(opt.norm_mean, opt.norm_std)
        ])
    else:
        raise RuntimeError('Error transformation mode.')