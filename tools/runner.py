import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc

import time
import os
import pickle


def convert_data_to_tensors(data_our, args):
    data = {}
    data['video'] = torch.cat([data_our[i]['video'].unsqueeze(0) for i in range(len(data_our))], 0)
    data_transits = [torch.from_numpy(data_our[i]['transits']) for i in range(len(data_our))]
    data['transits'] = torch.cat([data_transits[i].unsqueeze(0) for i in range(len(data_our))], 0)
    data_frame_labels = [torch.from_numpy(data_our[i]['frame_labels']) for i in range(len(data_our))]
    data['frame_labels'] = torch.cat([data_frame_labels[i].unsqueeze(0) for i in range(len(data_our))], 0)
    data['number'] = torch.tensor([data_our[i]['number'] for i in range(len(data_our))])
    data['final_score'] = torch.tensor([data_our[i]['final_score'] for i in range(len(data_our))])
    data['pred_info'] = torch.cat([data_our[i]['pred_info'].unsqueeze(0) for i in range(len(data_our))], 0)
    data['difficulty'] = torch.tensor([data_our[i]['difficulty'] for i in range(len(data_our))])
    data['completeness'] = torch.tensor([data_our[i]['completeness'] for i in range(len(data_our))])
    return data


def data_process_replace(data_all, args):
    target_list = [i[1] for i in data_all]
    data_our1 = [i[0] for i in data_all]
    for idx, targets in enumerate(target_list):
        if len(targets) < args.voter_number:
            filler = [targets[0]] * (args.voter_number - len(targets))
            target_list[idx] += filler
    data = convert_data_to_tensors(data_our1, args)
    target = []
    for i in range(args.voter_number):
        target_our = []
        for j in range(len(target_list)):
            target_our.append(target_list[j][i])
        target_index = convert_data_to_tensors(target_our, args)
        target.append(target_index)
    return data, target


def train_net(args):
    print('Trainer start ... ')
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                                   shuffle=True, num_workers=int(args.workers),
                                                   pin_memory=True, worker_init_fn=misc.worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test, collate_fn=lambda x: x,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)

    # build model
    PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose, i3d_dim_reducer, res_dim_reducer, learning_par = builder.model_builder(args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        PoseNet_model = PoseNet_model.cuda()
        base_model = base_model.cuda()
        I3D_encoder = I3D_encoder.cuda()
        psnet_model = psnet_model.cuda()
        decoder = [m.cuda() for m in decoder]
        decoder_pose = [m.cuda() for m in decoder_pose]
        regressor_delta = [m.cuda() for m in regressor_delta]
        i3d_dim_reducer = [m.cuda() for m in i3d_dim_reducer]
        res_dim_reducer = [m.cuda() for m in res_dim_reducer]
        learning_par = [m.cuda() for m in learning_par]
        torch.backends.cudnn.benchmark = True

    optimizer, scheduler = builder.build_opti_sche(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder,
                                                   regressor_delta, decoder_pose, i3d_dim_reducer,res_dim_reducer, learning_par, args)
    # print(scheduler)
    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    if torch.cuda.device_count() >1:
        PoseNet_model = nn.DataParallel(PoseNet_model)
        base_model = nn.DataParallel(base_model)
        I3D_encoder = nn.DataParallel(I3D_encoder)
        psnet_model = nn.DataParallel(psnet_model)
        decoder = [nn.DataParallel(_decoder) for _decoder in decoder]
        decoder_pose = [nn.DataParallel(_decoder_pose) for _decoder_pose in decoder_pose]
        regressor_delta = [nn.DataParallel(_regressor_delta) for _regressor_delta in regressor_delta]
        i3d_dim_reducer = [nn.DataParallel(_dim_reducer) for _dim_reducer in i3d_dim_reducer]
        res_dim_reducer = [nn.DataParallel(_dim_reducer) for _dim_reducer in res_dim_reducer]

    # loss
    mse = nn.MSELoss().cuda()
    ce = nn.CrossEntropyLoss().cuda()
    start_time = time.time()
    # training phase
    for epoch in range(start_epoch, args.max_epoch):
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []

        PoseNet_model.train()
        I3D_encoder.train()
        base_model.train()
        psnet_model.train()
        for _decoder in decoder:
            _decoder.train()
        for _decoder_pose in decoder_pose:
            _decoder_pose.train()
        for _regressor_delta in regressor_delta:
            _regressor_delta.train()
        for _dim_reducer in i3d_dim_reducer:
            _dim_reducer.train()
        for _dim_reducer in res_dim_reducer:
            _dim_reducer.train()

        for _learning_par in learning_par:
            _learning_par.train()
        if args.fix_bn:
            base_model.apply(misc.fix_bn)

        for idx, (data, target) in enumerate(train_dataloader):
            video_1 = data['video'].float().cuda()
            video_2 = target['video'].float().cuda()

            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas = target['transits'].float().cuda() + 1
            video_1_number = data['number'].float().cuda()
            video_2_number = target['number'].float().cuda()
            if args.using_DD:
                label_1_score = data['completeness'].float().reshape(-1, 1).cuda()
                label_2_score = target['completeness'].float().reshape(-1, 1).cuda()
            else:
                label_1_score = data['final_score'].float().reshape(-1, 1).cuda()
                label_2_score = target['final_score'].float().reshape(-1, 1).cuda()
            diff = data['difficulty'].float().reshape(-1, 1).cuda()
            pred_1 = data['pred_info'].float().cuda()
            pred_2 = target['pred_info'].float().cuda()
            helper.network_forward_train(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta,
                                         decoder_pose, i3d_dim_reducer,res_dim_reducer,learning_par, pred_scores,
                                         video_1,video_1_number, label_1_score, video_2,video_2_number, label_2_score, mse, optimizer,
                                         pred_1, pred_2, diff,
                                          epoch, idx + 1, len(train_dataloader),
                                         args, label_1_tas, label_2_tas, ce,
                                         pred_tious_5, pred_tious_75)
            true_scores.extend(data['final_score'].numpy())
            # break

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_mean_5 = sum(pred_tious_5) / len(train_dataset)
        pred_tious_mean_75 = sum(pred_tious_75) / len(train_dataset)

        print('[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f'
              % (epoch, pred_tious_mean_5, pred_tious_mean_75))

        print(
            '[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f' % (epoch, rho, L2, RL2,
                                                                                                    optimizer.param_groups[
                                                                                                        0]['lr'],
                                                                                                    optimizer.param_groups[
                                                                                                        1]['lr']))

        elapsed_time = time.time() - start_time
        epochs_completed = epoch + 1
        avg_time_per_epoch = elapsed_time / epochs_completed
        epochs_remaining = args.max_epoch - epochs_completed
        estimated_time_remaining = avg_time_per_epoch * epochs_remaining


        if (epoch >= 140) or (epoch < 140 and epoch % 2 == 0):
            validate(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose,
                     i3d_dim_reducer,res_dim_reducer,learning_par, test_dataloader, epoch, optimizer, args)
            helper.save_checkpoint(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta,
                                   decoder_pose, i3d_dim_reducer,res_dim_reducer,learning_par, optimizer, epoch,
                                   epoch_best_aqa, rho_best, L2_min, RL2_min, 'last', args)
            print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa,
                                                                                               rho_best, L2_min,
                                                                                               RL2_min))
            print('[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                               pred_tious_best_5, pred_tious_best_75))

        # scheduler lr
        if scheduler is not None:
            scheduler.step()


def validate(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose, i3d_dim_reducer,res_dim_reducer,learning_par,
             test_dataloader, epoch, optimizer, args):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min, epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    PoseNet_model.eval()
    base_model.eval()
    I3D_encoder.eval()
    psnet_model.eval()
    for _decoder in decoder:
        _decoder.eval()
    for _decoder_pose in decoder_pose:
        _decoder_pose.eval()
    for _regressor_delta in regressor_delta:
        _regressor_delta.eval()
    for _dim_reducer in i3d_dim_reducer:
        _dim_reducer.eval()
    for _dim_reducer in res_dim_reducer:
        _dim_reducer.eval()
    for _learning_par in learning_par:
        _learning_par.eval()
    val_num = 0
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, data_all in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            data, target = data_process_replace(data_all, args)
            val_num += len(data['final_score'])

            video_1 = data['video'].float().cuda()
            video_2_list = [item['video'].float().cuda() for item in target]
            diff = data['difficulty'].float().reshape(-1, 1).cuda()
            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas_list = [item['transits'].float().cuda() + 1 for item in target]

            video_2_number_list = [item['number'].float().cuda() for item in target]

            if args.using_DD:
                label_2_score_list = [item['completeness'].float().reshape(-1, 1).cuda() for item in target]
            else:
                label_2_score_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
            pred_1 = data['pred_info'].float().cuda()
            pred_2_list = [item['pred_info'].float().cuda() for item in target]


            loss_aqa = helper.network_forward_test(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder,
                                                   regressor_delta, decoder_pose, i3d_dim_reducer,res_dim_reducer,learning_par, pred_scores,
                                                   video_1, video_2_list,video_2_number_list, label_2_score_list,
                                                   pred_1, pred_2_list,
                                                   args, label_1_tas, label_2_tas_list, diff,
                                                   pred_tious_test_5, pred_tious_test_75)

            batch_time = time.time() - start
            if batch_idx % 2 == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f \t Aqa_loss %0.5f '
                      % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime, loss_aqa))
            datatime_start = time.time()
            if args.bs_test > 1:
                true_scores.extend(data['final_score'].numpy())
            else:
                true_scores.append(data['final_score'])
        # evaluation results

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (val_num * args.voter_number)
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (val_num * args.voter_number)

        if pred_tious_test_mean_5 > pred_tious_best_5:
            pred_tious_best_5 = pred_tious_test_mean_5
        if pred_tious_test_mean_75 > pred_tious_best_75:
            pred_tious_best_75 = pred_tious_test_mean_75
            epoch_best_tas = epoch
        print('[TEST] EPOCH: %d, tIoU_5: %.6f, tIoU_75: %.6f ,val_count:%d' % (
        epoch, pred_tious_test_mean_5, pred_tious_test_mean_75, val_num))

        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best_aqa = epoch
            print('-----New best found!-----')
            helper.save_outputs(pred_scores, true_scores, args)
            helper.save_checkpoint(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta,
                                   decoder_pose, i3d_dim_reducer,res_dim_reducer,learning_par, optimizer, epoch, epoch_best_aqa,
                                   rho_best, L2_min, RL2_min, 'best', args)
        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2))





