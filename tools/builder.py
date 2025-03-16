import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.optim as optim
import torch.nn as nn
from models import I3D_backbone
from models.PS import PSNet
from utils.misc import import_class
from torchvideotransforms import video_transforms, volume_transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ExponentialLR, CosineAnnealingLR

from models import decoder_fuser
from models import MLP_score, ScoreNet
from models.E2Emodel import E2EModel, SpotModel, EnhModel, C_channel

from models.MLP import LogitScaleNetwork as par
from models.PoseAttn import PoseAttnNet



def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((200, 112)),
        video_transforms.RandomCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((200, 112)),
        video_transforms.CenterCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def dataset_builder(args):
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    if args.test:
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return test_dataset
    else:
        train_dataset = Dataset(args, transform=train_trans, subset='train')
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return train_dataset, test_dataset


def model_builder(args):
    feat_dim = 256  # rny002 368  rny008/convnextt 768
    i3d_dim = 1024
    res_dim = 368
    classes = 2
    I3D_encoder = I3D_backbone(I3D_class=400)
    I3D_encoder.load_pretrain(args.pretrained_i3d_weight)
    base_model = E2EModel(args.feature_arch, clip_len=args.clip_len, modality=args.modality, )
    PSNet_model = SpotModel(classes, args.temporal_arch, feat_dim=res_dim)
    Model = import_class(args.model)
    PoseNet_model = Model(**args.model_args)
    i3d_dim_reducer = []
    res_dim_reducer = []
    decoder_pose = []
    Decoder_vit = []
    learning_par = []
    for i in range(args.step_num):
        i3d_dim_reducer.append(C_channel(fea_dim=i3d_dim))
        res_dim_reducer.append(C_channel(fea_dim=res_dim))
        decoder_pose.append(PoseAttnNet(dim=feat_dim, num_heads=args.num_heads, Pose_flag=args.Pose_flag))
        if i < 2:
            Decoder_vit.append(decoder_fuser(dim=feat_dim, num_heads=args.num_heads, num_layers=args.num_layers))
        learning_par.append(par())

    # Regressor_delta = MLP_score(in_channel=feat_dim, out_channel=1)
    Regressor_delta1 = MLP_score(in_channel=feat_dim, out_channel=1)
    Regressor_delta2 = MLP_score(in_channel=feat_dim, out_channel=1)

    return PoseNet_model, I3D_encoder, base_model, PSNet_model, Decoder_vit, [Regressor_delta1,
                                                                              Regressor_delta2], decoder_pose, i3d_dim_reducer, res_dim_reducer, learning_par


def build_opti_sche(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose,
                    dim_reducer,res_dim_reducer, learning_par, args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': PoseNet_model.parameters()},
            {'params': I3D_encoder.parameters()},
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': psnet_model.parameters()},
            {'params': decoder[0].parameters()},
            {'params': decoder[1].parameters()},
            {'params': decoder_pose[0].parameters()},
            {'params': decoder_pose[1].parameters()},
            {'params': decoder_pose[2].parameters()},
            {'params': regressor_delta[0].parameters()},
            {'params': regressor_delta[1].parameters()},
            {'params': dim_reducer[0].parameters()},
            {'params': dim_reducer[1].parameters()},
            {'params': dim_reducer[2].parameters()},
            {'params': res_dim_reducer[0].parameters()},
            {'params': res_dim_reducer[1].parameters()},
            {'params': res_dim_reducer[2].parameters()},
            {'params': learning_par[0].parameters()},
            {'params': learning_par[1].parameters()},
            {'params': learning_par[2].parameters()},
        ], lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    # scheduler = None
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    return optimizer, scheduler


def resume_train(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose, optimizer,
                 i3d_dim_reducer,res_dim_reducer, skeleton_dim_reducer,learning_par, args):
    ckpt_path = os.path.join(args.experiment_path, 'best.pth')  # 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model

    PoseNet_model_ckpt = {k.replace("module.", ""): v for k, v in state_dict['PoseNet_model'].items()}
    PoseNet_model.load_state_dict(PoseNet_model_ckpt)

    I3D_encoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['I3D_encoder'].items()}
    I3D_encoder.load_state_dict(I3D_encoder_ckpt)

    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_ckpt)
    for i in range(args.step_num):
        decoder_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'decoder_pose{i + 1}'].items()}
        decoder_pose[i].load_state_dict(decoder_pose_ckpt)
        i3d_dim_reducer_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'i3d_dim_reducer{i + 1}'].items()}
        i3d_dim_reducer[i].load_state_dict(i3d_dim_reducer_ckpt)
        res_dim_reducer_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'res_dim_reducer{i + 1}'].items()}
        res_dim_reducer[i].load_state_dict(res_dim_reducer_ckpt)
        skeleton_dim_reducer_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'skeleton_dim_reducer{i + 1}'].items()}
        skeleton_dim_reducer[i].load_state_dict(skeleton_dim_reducer_ckpt)
        learning_par_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'learning_par{i + 1}'].items()}
        learning_par[i].load_state_dict(learning_par_ckpt)


    for i in range(2):
        decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'decoder{i + 1}'].items()}
        decoder[i].load_state_dict(decoder_ckpt)
        regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'regressor_delta{i + 1}'].items()}
        regressor_delta[i].load_state_dict(regressor_delta_ckpt)


    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(PoseNet_model, I3D_encoder, base_model, psnet_model, decoder, regressor_delta, decoder_pose, dim_reducer,
               args):
    ckpt_path = args.ckpts
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    PoseNet_model_ckpt = {k.replace("module.", ""): v for k, v in state_dict['PoseNet_model'].items()}
    PoseNet_model.load_state_dict(PoseNet_model_ckpt)

    I3D_encoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['I3D_encoder'].items()}
    I3D_encoder.load_state_dict(I3D_encoder_ckpt)

    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_ckpt)

    for i in range(3):
        decoder_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'decoder_pose{i + 1}'].items()}
        decoder_pose[i].load_state_dict(decoder_pose_ckpt)

    for i in range(2):
        decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'decoder{i + 1}'].items()}
        decoder[i].load_state_dict(decoder_ckpt)

    for i in range(2):
        regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'regressor_delta{i + 1}'].items()}
        regressor_delta[i].load_state_dict(regressor_delta_ckpt)

    for i in range(3):
        dim_reducer_ckpt = {k.replace("module.", ""): v for k, v in state_dict[f'dim_reducer{i + 1}'].items()}
        dim_reducer[i].load_state_dict(dim_reducer_ckpt)

    epoch = state_dict['epoch']
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch, rho_best, L2_min, RL2_min))
    return


def load_trainmodel(PoseNet_model, base_model, psnet_model, decoder, regressor_delta, decoder_pose, args):
    ckpt_path = '/experiments/FineDiving/'
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')


    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)
    psnet_model_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_model_ckpt)
    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta'].items()}
    # print(state_dict['regressor_delta'].items())
    regressor_delta.load_state_dict(regressor_delta_ckpt)

    epoch = state_dict['epoch']
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch - 1, rho_best, L2_min, RL2_min))
    return
