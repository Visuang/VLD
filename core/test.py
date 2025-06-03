
import numpy as np
import torch
from torch.autograd import Variable
from tools import eval_regdb, eval_sysu,evaluate_vcm,MODALITY_,evaluate_bupt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def test(base, loader, config):
#     base.set_eval()
#     print('Extracting Query Feature...')
#     ptr = 0
#     query_feat = np.zeros((loader.n_query, 2048))
#     with torch.no_grad():
#         for batch_idx, (input, label) in enumerate(loader.query_loader):
#             batch_num = input.size(0)
#             input = Variable(input.cuda())
#             feat = base.ir_model(input)
#             feat = base.shared_model(feat)
#             feat = base.classifier(feat)
#             query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
#             ptr = ptr + batch_num
#
#     print('Extracting Gallery Feature...')
#
#     if loader.dataset == 'sysu':
#         all_cmc = 0
#         all_mAP = 0
#         all_mINP = 0
#         for i in range(10):
#             ptr = 0
#             gall_loader = loader.gallery_loaders[i]
#             gall_feat = np.zeros((loader.n_gallery, 768))
#             with torch.no_grad():
#                 for batch_idx, (input, label) in enumerate(gall_loader):
#                     batch_num = input.size(0)
#                     input = Variable(input.cuda())
#                     feat = base.rgb_model(input)
#                     feat = base.shared_model(feat)
#                     feat = base.classifier(feat)
#                     gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
#
#                     ptr = ptr + batch_num
#             distmat = np.matmul(query_feat, np.transpose(gall_feat))
#             cmc, mAP, mINP = eval_sysu(-distmat, loader.query_label, loader.gall_label, loader.query_cam,
#                                        loader.gall_cam)
#             all_cmc += cmc
#             all_mAP += mAP
#             all_mINP += mINP
#         all_cmc /= 10.0
#         all_mAP /= 10.0
#         all_mINP /= 10.0
#
#     elif loader.dataset == 'regdb':
#         gall_loader = loader.gallery_loaders
#         gall_feat = np.zeros((loader.n_gallery, 768))
#         ptr = 0
#         with torch.no_grad():
#             for batch_idx, (input, label) in enumerate(gall_loader):
#                 batch_num = input.size(0)
#                 input = Variable(input.cuda())
#                 feat = base.rgb_model(input)
#                 feat = base.shared_model(feat)
#                 feat = base.classifier(feat)
#                 gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
#
#                 ptr = ptr + batch_num
#         if config.regdb_test_mode == 't-v':
#             distmat = np.matmul(query_feat, np.transpose(gall_feat))
#             cmc, mAP, mINP = eval_regdb(-distmat, loader.query_label, loader.gall_label)
#         else:
#             distmat = np.matmul(gall_feat, np.transpose(query_feat))
#             cmc, mAP, mINP = eval_regdb(-distmat, loader.gall_label, loader.query_label)
#
#         all_cmc, all_mAP, all_mINP = cmc, mAP, mINP
#
#
#     return all_cmc, all_mAP, all_mINP


def process_vedio(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def tem_pool(features,t=6):
    features = features.squeeze()
    features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
    features = features.mean(0)
    return features


def test_vcm(base, loader):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []

    query_loader = loader.query_loader
    query_feat = np.zeros((loader.samples.num_query_tracklets, 1536))

    with torch.no_grad():
        for batch_idx, (input, label,c_label) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            input = Variable(input.cuda())
            input = process_vedio(input)
            feat = base.model(x1=input)
            query = feat

            query_feat[ptr:ptr + batch_num, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'vcm':
        g_pids = []
        g_camids = []

        ptr = 0

        gall_loader = loader.gallery_loader
        gall_feat = np.zeros((loader.samples.num_gallery_tracklets, 1536))


        with torch.no_grad():
            for batch_idx, (input, label,c_label) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                input = Variable(input.cuda())
                input = process_vedio(input)
                feat = base.model(x1=input)
                gall = feat

                gall_feat[ptr:ptr + batch_num, :] = gall.detach().cpu().numpy()
                ptr = ptr + batch_num

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc, mAP = evaluate_vcm(-distmat, q_pids, g_pids, q_camids,
                                  g_camids)

        all_cmc1  = cmc
        all_mAP1  = mAP


        distmat = distmat.T
        cmc, mAP = evaluate_vcm(-distmat,  g_pids,q_pids,
                                  g_camids, q_camids)

        all_cmc2  = cmc
        all_mAP2  = mAP


    str1 = print_metrics(
        all_cmc1, all_mAP1,
        prefix='{:<3}->{:<3}:  '.format('IR', 'RGB')
    )

    str2 = print_metrics(
        all_cmc2, all_mAP2,
        prefix='{:<3}->{:<3}:  '.format('RGB', 'IR')
    )
    return all_cmc1, all_mAP1,all_cmc2, all_mAP2, str1,str2


def print_metrics(cmc, ap, prefix=''):
    str = '{}mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}.'.format(prefix, ap, cmc[0], cmc[4], cmc[9], cmc[19])
    return str

def test_bupt(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []
    q_modalitys = []

    query_loader = loader.query_loader
    query_feat = np.zeros((1048, 1536))


    with torch.no_grad():
        for batch_idx, (input, label,c_label,modals) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            q_modalitys.extend(modals)
            input = Variable(input.cuda())
            input = process_vedio(input)
            if modals[0]==0:
                feat = base.model(x1=input)
            elif modals[0]==1:
                feat = base.model(x1=input)
            else:
                continue

            query = feat

            query_feat[ptr:ptr + batch_num, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_modalitys = np.asarray(q_modalitys)
        # query_feat = torch.cat(query_feats, dim=0)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'bupt':
        g_pids = []
        g_camids = []
        g_modalitys = []
        ptr = 0

        gall_loader = loader.gallery_loader
        gall_feat = np.zeros((4790, 1536))


        with torch.no_grad():
            for batch_idx, (input, label,c_label,modals) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                g_modalitys.extend(modals)
                input = Variable(input.cuda())
                input = process_vedio(input)
                if modals[0]==0:
                    feat = base.model(x1=input)
                elif modals[1]==1:
                    feat = base.model(x1=input)
                else:
                    continue
                gall = feat
                gall_feat[ptr:ptr + batch_num, :] = gall.detach().cpu().numpy()
                ptr = ptr + batch_num

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_modalitys = np.asarray(g_modalitys)
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        # evaluate (intra/inter-modality)
        CMC, MAP = [], []
        eval_str = []
        for q_modal in (0, 1):
            for g_modal in (0, 1):
                if q_modal!=g_modal:
                    q_mask = q_modalitys == q_modal
                    g_mask = g_modalitys == g_modal
                    tmp_distance = distmat[q_mask, :][:, g_mask]
                    tmp_qid = q_pids[q_mask]
                    tmp_gid = g_pids[g_mask]
                    tmp_cmc, tmp_ap = evaluate_bupt(-tmp_distance, tmp_qid, tmp_gid, config)
                    CMC.append(tmp_cmc * 100)
                    MAP.append(tmp_ap * 100)

                    str = print_metrics(
                        tmp_cmc, tmp_ap,
                        prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
                    )
                    eval_str.append(str)

    return CMC, MAP,eval_str