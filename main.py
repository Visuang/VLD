
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import ast
import torch
import random
import argparse
import numpy as np
from torch.cuda import amp
from data_loader.loader import Loader
from core import Base, train, test,test_vcm,test_bupt
from tools import make_dirs, Logger, os_walk, time_now
import warnings
warnings.filterwarnings("ignore")

import setproctitle

# 设置进程名
setproctitle.setproctitle("李爽")

best_mAP = 0
best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(config):
    global best_mAP
    global best_rank1

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    if config.mode == 'train':
        start_train_epoch = 0


        scaler = amp.GradScaler()
        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            model.model_lr_scheduler.step(current_epoch)

            if current_epoch < config.total_train_epoch:
                _, result = train(model, loaders, config,scaler)
                logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), current_epoch, result,model.model_optimizer.state_dict()['param_groups'][0]['lr']))

            if current_epoch + 1 >= 0 and (current_epoch + 1) % config.eval_epoch == 0:
                if config.dataset=='bupt':
                    cmc, mAP,eval_str = test_bupt(model, loaders, config)
                    is_best_rank = (cmc[1][0] >= best_rank1)
                    best_rank1 = max(cmc[1][0], best_rank1)
                    model.save_model(current_epoch, is_best_rank)
                    logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))
                elif config.dataset=='vcm':
                    cmc, mAP,cmc2, mAP2, eval_str_t2v,eval_str_v2t = test_vcm(model, loaders)
                    is_best_rank = (cmc[0] >= best_rank1)
                    best_rank1 = max(cmc[0], best_rank1)
                    model.save_model(current_epoch, is_best_rank)
                    logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str_t2v,eval_str_v2t))

    elif config.mode == 'test':
        model.resume_model(config.resume_test_path)
        if config.dataset == 'bupt':
            cmc, mAP, eval_str = test_bupt(model, loaders, config)
            logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                    eval_str[0], eval_str[1]))
        elif config.dataset == 'vcm':
            cmc, mAP, cmc2, mAP2, eval_str_t2v, eval_str_v2t = test_vcm(model, loaders)
            logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                    eval_str_t2v, eval_str_v2t))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    parser.add_argument('--dataset', default='vcm', help='dataset name: vcm , bupt]')
    parser.add_argument('--vcm_data_path', type=str, default='/data1/ls/data/HITSZ-VCM/')
    parser.add_argument('--bupt_data_path', type=str, default='/data1/ls/data/BUPTCampus/')
    parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--is_STA',type=ast.literal_eval,default=True)
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pid_num', type=int, default=1074) #bupt 1074
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.000025)
    parser.add_argument('--lr_times', type=int, default=25)
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--output_path', type=str, default='vcm/debug/VLD_vcm',
                        help='path to save related informations')
    parser.add_argument('--total_train_epoch', type=int, default=24)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--resume_test_path', type=str, default='/data1/ls/code/VIV_ReID/Github_vld/VLD/bupt/debug/VLD_vcm/models')
    parser.add_argument('--STH_start_layer', type=int, default=9, help='-1 for no resuming')

    ###bupt
    parser.add_argument('--train_frame_sample', type=str, default='random')
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--random_flip', action='store_false', default=True)
    parser.add_argument('--fake', action='store_true', default=False)
    parser.add_argument('--test_frame_sample', type=str, default='uniform')
    parser.add_argument('--test_sampler', type=str,
                             default='ConsistentModalitySampler', help='None for no shuffle')
    parser.add_argument('--test_bs', type=int, default=64)  # Please don't change it.
    parser.add_argument('--train_sampler', type=str,
                             default='RandomIdentitySampler', help='None for shuffle')
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--train_sampler_nc', type=int, default=2)
    parser.add_argument('--train_sampler_nt', type=int, default=1)
    parser.add_argument('--max_rank', type=int, default=20)

    #loss
    parser.add_argument('--weight_a', type=float, default=0.08)
    parser.add_argument('--weight_b', type=float, default=0.4)
    parser.add_argument('--weight_c', type=float, default=1.0)
    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)
