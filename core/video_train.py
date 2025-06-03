import torch
import torch.nn.functional as F
import copy
from torch.cuda import amp
from tools.loss import diversity_loss

def process_vedio(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def foward_video(iter,base,meter,scaler):
    # scaler = amp.GradScaler()
    for _ in range(base.steps):
        input1, input2, label1, label2 = iter.next_one()
        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        rgb_imgs = process_vedio(rgb_imgs,base.seq_lenth)
        ir_imgs = process_vedio(ir_imgs,base.seq_lenth)

        with amp.autocast(enabled=True):
            features, cls_score = base.model(x1=rgb_imgs, x2=ir_imgs)

            pids = torch.cat([rgb_pids, ir_pids], dim=0)

            ide_loss = base.pid_creiteron(cls_score[0], pids)
            ide_loss_proj = base.pid_creiteron(cls_score[1], pids)
            ide_loss_text = base.pid_creiteron(cls_score[2], pids)
            ide_loss_cue = base.pid_creiteron(cls_score[3], pids)


            triplet_loss_last = base.tri_creiteron(features[0].squeeze(), pids)
            triplet_loss = base.tri_creiteron(features[1].squeeze(), pids)
            triplet_loss_proj = base.tri_creiteron(features[2].squeeze(), pids)
            triplet_loss_cue = base.tri_creiteron(features[3].squeeze(), pids)
            ide_loss_text = base.weight_a*ide_loss_text
            ide_loss_cue = base.weight_b*ide_loss_cue
            triplet_loss_cue = base.weight_c*triplet_loss_cue


            total_loss = ide_loss + ide_loss_proj + triplet_loss_last + triplet_loss + triplet_loss_proj + ide_loss_text + ide_loss_cue + triplet_loss_cue
        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()




        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'tri_loss': triplet_loss.data,
                      'tri_loss_proj': triplet_loss_proj.data,
                      'tri_loss_last': triplet_loss_last.data,
                      'ide_loss_text': ide_loss_text.data,
                      'pid_loss_STP': ide_loss_cue.data,
                      'tri_loss_STP': triplet_loss_cue.data,
                      })
    return meter