import torch
from tools import MultiItemAverageMeter
from core.video_train import foward_video
from torch.cuda import amp

def train(base, loaders, config,scaler):

    base.set_train()
    meter = MultiItemAverageMeter()
    if config.dataset=='vcm':
        loader = loaders.get_train_loader()
    elif config.dataset=='bupt':
        loader = loaders.train_iter
    # if config.module == 'video_B_tri':
    meter = foward_video(loader,base,meter,scaler)




    return meter.get_val(), meter.get_str()







