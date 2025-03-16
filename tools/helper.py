import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.nn as nn
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d
import torch.nn.functional as F


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())



def contrastive_loss_with_sim_and_tau(A, B, fix_size, tau=0.1):

    stage_A = torch.stack([torch.mean(A[0:1*fix_size], dim=0),
                           torch.mean(A[1*fix_size:2*fix_size], dim=0),
                           torch.mean(A[2*fix_size:3*fix_size], dim=0)])
    stage_B = torch.stack([torch.mean(B[0:1*fix_size], dim=0),
                           torch.mean(B[1*fix_size:2*fix_size], dim=0),
                           torch.mean(B[2*fix_size:3*fix_size], dim=0)])

    sim_AA = sim(stage_A, stage_A)
    sim_AB = sim(stage_A, stage_B)
    f = lambda x: torch.exp(x / tau)

    loss = 0.0
    for i in range(3):
        pos_similarity = f(sim_AB[i, i])
        neg_similarity_A = sum([f(sim_AA[i, j]) for j in range(3) if j != i])
        cross_neg_similarity = sum([f(sim_AB[i, j]) for j in range(3) if j != i])
        Z = pos_similarity + neg_similarity_A + cross_neg_similarity
        normalized_pos = pos_similarity / Z
        loss += -torch.log(normalized_pos)

    return loss / 3



def Multi_Stage_Seg_train(args, i3d_fea_1, res_fea_1, label_1_tas, skeleton_fea_1, i3d_fea_2, res_fea_2,
                          label_2_tas, skeleton_fea_2):
    i3d_1_segs = []
    res_1_segs = []
    skeleton_1_segs = []
    for bs_1 in range(i3d_fea_1.shape[0]):
        i3d_1_st = int(label_1_tas[bs_1][0].item()) // 10
        i3d_1_ed = int(label_1_tas[bs_1][1].item()) // 10
        res_1_st = int(label_1_tas[bs_1][0].item())
        res_1_ed = int(label_1_tas[bs_1][1].item())
        sk_1_st = int(label_1_tas[bs_1][0].item())
        sk_1_ed = int(label_1_tas[bs_1][1].item())
        i3d_1_segs.append(
            seg_pool_1d(i3d_fea_1[bs_1].unsqueeze(0), i3d_1_st, i3d_1_ed, args.fix_size))
        res_1_segs.append(
            seg_pool_1d(res_fea_1[bs_1].unsqueeze(0), res_1_st, res_1_ed, args.fix_size))
        skeleton_1_segs.append(
            seg_pool_1d(skeleton_fea_1[bs_1].unsqueeze(0), sk_1_st, sk_1_ed, args.fix_size))
    i3d_1_segs = torch.cat(i3d_1_segs, 0).transpose(1, 2)
    res_1_segs = torch.cat(res_1_segs, 0).transpose(1, 2)
    skeleton_1_segs = torch.cat(skeleton_1_segs, 0).transpose(1, 2)
    # reshape(batch, joint, h, w, -1).permute(0, 1, 4, 2, 3)

    i3d_2_segs = []
    res_2_segs = []
    skeleton_2_segs = []
    for bs_2 in range(i3d_fea_2.shape[0]):
        i3d_2_st = int(label_2_tas[bs_2][0].item()) // 10
        i3d_2_ed = int(label_2_tas[bs_2][1].item()) // 10
        res_2_st = int(label_2_tas[bs_2][0].item())
        res_2_ed = int(label_2_tas[bs_2][1].item())
        sk_2_st = int(label_2_tas[bs_2][0].item())
        sk_2_ed = int(label_2_tas[bs_2][1].item())
        i3d_2_segs.append(
            seg_pool_1d(i3d_fea_2[bs_2].unsqueeze(0), i3d_2_st, i3d_2_ed, args.fix_size))
        res_2_segs.append(
            seg_pool_1d(res_fea_2[bs_2].unsqueeze(0), res_2_st, res_2_ed, args.fix_size))
        skeleton_2_segs.append(
            seg_pool_1d(skeleton_fea_2[bs_2].unsqueeze(0), sk_2_st, sk_2_ed, args.fix_size))
    i3d_2_segs = torch.cat(i3d_2_segs, 0).transpose(1, 2)
    res_2_segs = torch.cat(res_2_segs, 0).transpose(1, 2)
    skeleton_2_segs = torch.cat(skeleton_2_segs, 0).transpose(1, 2)
    return [i3d_1_segs, res_1_segs, skeleton_1_segs], [i3d_2_segs, res_2_segs, skeleton_2_segs]





def network_forward_train(PoseNet_model, I3D_encoder, base_model, PSNet_model, decoder, regressor_delta, decoder_pose,
                          i3d_dim_reducer,res_dim_reducer,learning_par, pred_scores,
                          video_1, video_1_number, label_1_score, video_2, video_2_number, label_2_score, mse,
                          optimizer,
                          pose_1, pose_2,  diff,
                          epoch, batch_idx, batch_num, args, label_1_tas, label_2_tas, ce,
                          pred_tious_5, pred_tious_75):
    start = time.time()
    optimizer.zero_grad()
    com_feature_12, com_feamap_12 = I3D_encoder(video_1, video_2)  # com_feature :[16,9,2048]
    video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]  # video_1_fea:[16,9,1024]
    video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]
    i3d_fea = torch.cat((video_1_fea, video_2_fea), 0)
    total_video = torch.cat((video_1, video_2), 0)
    res_fea = base_model(total_video)
    transits_pred = PSNet_model(res_fea)
    total_pred = torch.cat((pose_1, pose_2), 0)
    number_pred, total_pred = PoseNet_model(total_pred)
    label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)  # 2N,2
    label_12_pad = torch.zeros(transits_pred.size())  # 16, 96, 2
    for bs in range(transits_pred.shape[0]):
        label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
        label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 2
    label_12_number = torch.cat((video_1_number, video_2_number), 0)
    loss_tas = ce(transits_pred, label_12_pad.cuda())
    loss_number = ce(number_pred, label_12_number.long())
    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num

    i3d_fea_1 = i3d_fea[:i3d_fea.shape[0] // 2].transpose(1, 2)
    i3d_fea_2 = i3d_fea[i3d_fea.shape[0] // 2:].transpose(1, 2)  # 8, dim, 96
    res_fea_1 = res_fea[:res_fea.shape[0] // 2].transpose(1, 2)
    res_fea_2 = res_fea[res_fea.shape[0] // 2:].transpose(1, 2)  # 8, dim, 96
    pred_fea_96_1 = total_pred[:total_pred.shape[0] // 2]
    pred_fea_96_2 = total_pred[total_pred.shape[0] // 2:]

    skeleton_fea_1 = pred_fea_96_1
    skeleton_fea_2 = pred_fea_96_2
    segs1, segs2 = Multi_Stage_Seg_train(args, i3d_fea_1, res_fea_1, label_1_tas, skeleton_fea_1, i3d_fea_2,
                                         res_fea_2, label_2_tas, skeleton_fea_2)
    i3d_1_segs, res_1_segs, skeleton_1_segs = segs1[0], segs1[1], segs1[2]  # video_1_segs : [16,15,1024]
    i3d_2_segs, res_2_segs, skeleton_2_segs = segs2[0], segs2[1], segs2[2]

    i3d_1_segs_fea = []
    i3d_2_segs_fea = []
    res_1_segs_fea = []
    res_2_segs_fea = []

    for i in range(args.step_num):
        i3d_1_segs_fea.append(i3d_dim_reducer[i](i3d_1_segs[:, args.fix_size * i :args.fix_size *(i+1), :]))
        i3d_2_segs_fea.append(i3d_dim_reducer[i](i3d_2_segs[:, args.fix_size * i :args.fix_size *(i+1), :]))
        res_1_segs_fea.append(res_dim_reducer[i](res_1_segs[:, args.fix_size * i :args.fix_size *(i+1), :]))
        res_2_segs_fea.append(res_dim_reducer[i](res_2_segs[:, args.fix_size * i :args.fix_size *(i+1), :]))

    i3d_1_segs_fea = torch.cat(i3d_1_segs_fea,dim=1)
    i3d_2_segs_fea = torch.cat(i3d_2_segs_fea,dim=1)
    res_1_segs_fea = torch.cat(res_1_segs_fea,dim=1)
    res_2_segs_fea = torch.cat(res_2_segs_fea,dim=1)

    #### Pose-Guided Attention ###########
    feat_pose_1_map_list = []
    feat_pose_2_map_list = []
    for i in range(args.step_num):
        feat_pose_1_map = decoder_pose[i](i3d_1_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          skeleton_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :])
        feat_pose_2_map = decoder_pose[i](i3d_2_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          skeleton_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :])
        feat_pose_1_map_list.append(feat_pose_1_map)
        feat_pose_2_map_list.append(feat_pose_2_map)

    video_1_segs = torch.cat(feat_pose_1_map_list, 1)
    video_2_segs = torch.cat(feat_pose_2_map_list, 1)

    ret_list_i3d = []
    ret_list_res = []
    for bs in range(i3d_1_segs_fea.shape[0]):
        l1 = contrastive_loss_with_sim_and_tau(video_1_segs[bs],
                                               video_2_segs[bs], args.fix_size)
        l2 = contrastive_loss_with_sim_and_tau(video_2_segs[bs],
                                               video_1_segs[bs], args.fix_size)
        l3 = contrastive_loss_with_sim_and_tau(res_1_segs_fea[bs],
                                               res_2_segs_fea[bs], args.fix_size)
        l4 = contrastive_loss_with_sim_and_tau(res_2_segs_fea[bs],
                                               res_1_segs_fea[bs], args.fix_size)
        ret_i3d = (l1 + l2) * 0.5
        ret_list_i3d.append(ret_i3d)
        ret_res = (l3 + l4) * 0.5
        ret_list_res.append(ret_res)
    loss_cont_i3d = sum(ret_list_i3d)/len(ret_list_i3d) * 10
    loss_cont_res = sum(ret_list_res)/len(ret_list_res) * 10

    decoder_video_12_map_list = []
    decoder_video_21_map_list = []
    decoder_video_12_res_list = []
    decoder_video_21_res_list = []
    for i in range(args.step_num):
        decoder_video_12_map = decoder[0](video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          )  # N,15,64
        decoder_video_21_map = decoder[0](video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          )  # N,15,64
        decoder_video_12_res = decoder[1](res_1_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          res_2_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          )  # N,15,64
        decoder_video_21_res = decoder[1](res_2_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          res_1_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                          )  # N,15,64
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)
        decoder_video_12_res_list.append(decoder_video_12_res)
        decoder_video_21_res_list.append(decoder_video_21_res)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)
    decoder_video_12_res = torch.cat(decoder_video_12_res_list, 1)
    decoder_video_21_res = torch.cat(decoder_video_21_res_list, 1)
    decoder_video_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
    decoder_video_res_12_21 = torch.cat((decoder_video_12_res, decoder_video_21_res), 0)

    # ############ Fine-grained Contrastive Regression #############
    delta1 = regressor_delta[0](decoder_video_12_21)
    delta2 = regressor_delta[1](decoder_video_res_12_21)
    delta1_1 = delta1[:, :args.fix_size].mean(1)
    delta1_2 = delta1[:, args.fix_size:2*args.fix_size].mean(1)
    delta1_3 = delta1[:, 2*args.fix_size:3*args.fix_size].mean(1)

    delta2_1 = delta2[:, :args.fix_size].mean(1)
    delta2_2 = delta2[:, args.fix_size:2*args.fix_size].mean(1)
    delta2_3 = delta2[:, 2*args.fix_size:3*args.fix_size].mean(1)
    delta1 = (delta1_1 * learning_par[0] + delta1_2 * learning_par[1] + delta1_3 * learning_par[2]) / (learning_par[0] + learning_par[1] + learning_par[2])
    delta2 = (delta2_1 * learning_par[0] + delta2_2 * learning_par[1] + delta2_3 * learning_par[2]) / (learning_par[0] + learning_par[1] + learning_par[2])
    delta = torch.cat((delta1, delta2), 1)
    delta = delta.mean(1).unsqueeze(-1)
    loss_aqa = mse(delta[:delta.shape[0] // 2], (label_1_score - label_2_score)) \
               + mse(delta[delta.shape[0] // 2:], (label_2_score - label_1_score))
    loss = loss_aqa + loss_tas + loss_cont_i3d + loss_cont_res
    loss.backward()
    optimizer.step()

    end = time.time()
    batch_time = end - start
    if args.using_DD:
        score = (delta[:delta.shape[0] // 2].detach() + label_2_score) * diff
    else:
        score = (delta[:delta.shape[0] // 2].detach() + label_2_score)
    pred_scores.extend([i.item() for i in score])
    tIoU_results = []
    for bs in range(transits_pred.shape[0] // 2):
        tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                        np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                        args))

    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    Batch_tIoU_5 = tIoU_correct_per_thr[0]
    Batch_tIoU_75 = tIoU_correct_per_thr[1]
    pred_tious_5.extend([Batch_tIoU_5])
    pred_tious_75.extend([Batch_tIoU_75])
    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t '
              'lr1 : %0.5f \t lr2 : %0.5f'
              '\t l_aqa : %0.5f \t l_ce : %0.5f \t loss_cont_i3d : %0.5f \t loss_cont_res : %0.5f \t  loss_number : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, loss.item(),
                 optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                 loss_aqa, loss_tas, loss_cont_i3d, loss_cont_res, loss_number))


def network_forward_test(PoseNet_model, I3D_encoder, base_model, PSNet_model, decoder, regressor_delta, decoder_pose,
                         i3d_dim_reducer,res_dim_reducer,learning_par, pred_scores,
                         video_1,  video_2_list, video_2_number_list, label_2_score_list,
                         pred_1, pred_2_list,
                         args, label_1_tas, label_2_tas_list, diff,
                         pred_tious_test_5, pred_tious_test_75):
    score = 0
    tIoU_results = []
    loss_aqa = 0

    for video_2, video_2_number, label_2_score, label_2_tas, pred_2 in zip(video_2_list, video_2_number_list,
                                                                                   label_2_score_list, label_2_tas_list,
                                                                                   pred_2_list):

        com_feature_12, com_feamap_12 = I3D_encoder(video_1, video_2)
        video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]
        video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]
        i3d_fea = torch.cat((video_1_fea, video_2_fea), 0)
        total_video = torch.cat((video_1, video_2), 0)
        res_fea = base_model(total_video)
        transits_pred = PSNet_model(res_fea)
        total_pred = torch.cat((pred_1, pred_2), 0)
        number_pred, total_pred = PoseNet_model(total_pred)

        label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)
        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = torch.zeros(label_12_tas.size())
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(
                    0).cpu().item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]
        i3d_fea_1 = i3d_fea[:i3d_fea.shape[0] // 2].transpose(1, 2)
        i3d_fea_2 = i3d_fea[i3d_fea.shape[0] // 2:].transpose(1, 2)  # 8, dim, 96
        res_fea_1 = res_fea[:res_fea.shape[0] // 2].transpose(1, 2)
        res_fea_2 = res_fea[res_fea.shape[0] // 2:].transpose(1, 2)  # 8, dim, 96
        pred_fea_96_1 = total_pred[:total_pred.shape[0] // 2]
        pred_fea_96_2 = total_pred[total_pred.shape[0] // 2:]

        skeleton_fea_1 = pred_fea_96_1
        skeleton_fea_2 = pred_fea_96_2
        segs1, segs2 = Multi_Stage_Seg_train(args, i3d_fea_1, res_fea_1, label_1_tas_pred, skeleton_fea_1, i3d_fea_2,
                                             res_fea_2, label_2_tas_pred, skeleton_fea_2)

        i3d_1_segs, res_1_segs, skeleton_1_segs = segs1[0], segs1[1], segs1[2]  # video_1_segs : [16,15,1024]
        i3d_2_segs, res_2_segs, skeleton_2_segs = segs2[0], segs2[1], segs2[2]

        i3d_1_segs_fea = []
        i3d_2_segs_fea = []
        res_1_segs_fea = []
        res_2_segs_fea = []

        for i in range(args.step_num):
            i3d_1_segs_fea.append(i3d_dim_reducer[i](i3d_1_segs[:, args.fix_size * i:args.fix_size * (i + 1), :]))
            i3d_2_segs_fea.append(i3d_dim_reducer[i](i3d_2_segs[:, args.fix_size * i:args.fix_size * (i + 1), :]))
            res_1_segs_fea.append(res_dim_reducer[i](res_1_segs[:, args.fix_size * i:args.fix_size * (i + 1), :]))
            res_2_segs_fea.append(res_dim_reducer[i](res_2_segs[:, args.fix_size * i:args.fix_size * (i + 1), :]))

        i3d_1_segs_fea = torch.cat(i3d_1_segs_fea, dim=1)
        i3d_2_segs_fea = torch.cat(i3d_2_segs_fea, dim=1)
        res_1_segs_fea = torch.cat(res_1_segs_fea, dim=1)
        res_2_segs_fea = torch.cat(res_2_segs_fea, dim=1)
        #### Pose-Guided Attention ###########
        feat_pose_1_map_list = []
        feat_pose_2_map_list = []
        for i in range(args.step_num):
            feat_pose_1_map = decoder_pose[i](i3d_1_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              skeleton_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :])
            feat_pose_2_map = decoder_pose[i](i3d_2_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              skeleton_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :])
            feat_pose_1_map_list.append(feat_pose_1_map)
            feat_pose_2_map_list.append(feat_pose_2_map)

        video_1_segs = torch.cat(feat_pose_1_map_list, 1)
        video_2_segs = torch.cat(feat_pose_2_map_list, 1)

        ############# Procedure-aware Cross-attention #############
        decoder_video_12_map_list = []
        decoder_video_21_map_list = []
        decoder_video_12_res_list = []
        decoder_video_21_res_list = []
        for i in range(args.step_num):
            decoder_video_12_map = decoder[0](video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              )  # N,15,64
            decoder_video_21_map = decoder[0](video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              )  # N,15,64
            decoder_video_12_res = decoder[1](res_1_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              res_2_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              )  # N,15,64
            decoder_video_21_res = decoder[1](res_2_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              res_1_segs_fea[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              )  # N,15,64
            decoder_video_12_map_list.append(decoder_video_12_map)
            decoder_video_21_map_list.append(decoder_video_21_map)
            decoder_video_12_res_list.append(decoder_video_12_res)
            decoder_video_21_res_list.append(decoder_video_21_res)

        decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
        decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)
        decoder_video_12_res = torch.cat(decoder_video_12_res_list, 1)
        decoder_video_21_res = torch.cat(decoder_video_21_res_list, 1)
        decoder_video_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
        decoder_video_res_12_21 = torch.cat((decoder_video_12_res, decoder_video_21_res), 0)

        ############ Fine-grained Contrastive Regression #############

        delta1 = regressor_delta[0](decoder_video_12_21)
        delta2 = regressor_delta[1](decoder_video_res_12_21)
        delta1_1 = delta1[:, :args.fix_size].mean(1)
        delta1_2 = delta1[:, args.fix_size:2 * args.fix_size].mean(1)
        delta1_3 = delta1[:, 2 * args.fix_size:3 * args.fix_size].mean(1)

        delta2_1 = delta2[:, :args.fix_size].mean(1)
        delta2_2 = delta2[:, args.fix_size:2 * args.fix_size].mean(1)
        delta2_3 = delta2[:, 2 * args.fix_size:3 * args.fix_size].mean(1)
        delta1 = (delta1_1 * learning_par[0] + delta1_2 * learning_par[1] + delta1_3 * learning_par[2]) / (
                    learning_par[0] + learning_par[1] + learning_par[2])
        delta2 = (delta2_1 * learning_par[0] + delta2_2 * learning_par[1] + delta2_3 * learning_par[2]) / (
                    learning_par[0] + learning_par[1] + learning_par[2])
        delta = torch.cat((delta1, delta2), 1)
        delta = delta.mean(1).unsqueeze(-1)
        if args.using_DD:
            score += (delta[:delta.shape[0] // 2].detach() + label_2_score) * diff
        else:
            score += (delta[:delta.shape[0] // 2].detach() + label_2_score)

        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))


    pred_scores.extend([i.item() / len(video_2_list) for i in score])
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])


    return loss_aqa / len(video_2_list)


def save_checkpoint(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose,
                    i3d_dim_reducer,res_dim_reducer,learning_par, optimizer, epoch,
                    epoch_best_aqa, rho_best, L2_min, RL2_min, prefix, args):
    torch.save({
        'PoseNet_model': PoseNet_model.state_dict(),
        'I3D_encoder': I3D_encoder.state_dict(),
        'base_model': base_model.state_dict(),
        'psnet_model': psnet_model.state_dict(),
        'decoder1': decoder[0].state_dict(),
        'decoder2': decoder[1].state_dict(),
        # 'decoder3': decoder[2].state_dict(),
        'regressor_delta1': regressor_delta[0].state_dict(),
        'regressor_delta2': regressor_delta[1].state_dict(),
        'decoder_pose1': decoder_pose[0].state_dict(),
        'decoder_pose2': decoder_pose[1].state_dict(),
        'decoder_pose3': decoder_pose[2].state_dict(),
        'i3d_dim_reducer1': i3d_dim_reducer[0].state_dict(),
        'i3d_dim_reducer2': i3d_dim_reducer[1].state_dict(),
        'i3d_dim_reducer3': i3d_dim_reducer[2].state_dict(),
        'res_dim_reducer1': res_dim_reducer[0].state_dict(),
        'res_dim_reducer2': res_dim_reducer[1].state_dict(),
        'res_dim_reducer3': res_dim_reducer[2].state_dict(),
        'learning_par1': learning_par[0].state_dict(),
        'learning_par2': learning_par[1].state_dict(),
        'learning_par3': learning_par[2].state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }, os.path.join(args.experiment_path, prefix + '.pth'))


def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
