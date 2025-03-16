import os
import yaml
import argparse
from torchlight import DictAction
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type = str, choices=['FineDiving'], help = 'dataset')
    parser.add_argument('--prefix', type = str, default='default', help = 'experiment name')
    parser.add_argument('--resume', action='store_true', default=False ,help = 'resume training (interrupted by accident)')
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--fix_bn', type=bool, default=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type=str, default="", help='test used ckpt path')
    parser.add_argument('--Heat_flag',type=bool,default=False)
    parser.add_argument('--using_DD',type=bool,default=False)
    parser.add_argument('--num_heads',type=int,default=False)
    parser.add_argument('--num_layers',type=int,default=False)
    parser.add_argument('--Pose_flag',type=bool,default=False)
    parser.add_argument(
        '--feature_arch', type=str, choices=[
            # From torchvision
            'rn18',
            'rn18_tsm',
            'rn18_gsm',
            'rn34_gsm',
            'rn50',
            'rn50_tsm',
            'rn50_gsm',
            # From timm (following its naming conventions)
            'rny002',
            'rny002_tsm',
            'rny002_gsm',
            'rny008',
            'rny008_tsm',
            'rny008_gsm',
            # From timm
            'convnextt',
            'convnextt_tsm',
            'convnextt_gsm'
        ], default='rny002_gsm', help='CNN architecture for feature extraction')
    parser.add_argument(
        '--temporal_arch', type=str, default='gru',
        choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
        help='Spotting architecture, after spatial pooling')
    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--modality', type=str, choices=['rgb', 'bw', 'flow'],
                        default='rgb')
    parser.add_argument('-mgpu', '--gpu_parallel', default=False, action='store_true')
    # GCN
    parser.add_argument('--GCN-config', type=str, default='config/bone_com_1.yaml')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    # optim
    parser.add_argument(
        '--gcn-base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--gcn-weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-ratio',
        type=float,
        default=0.001,
        help='decay rate for learning rate')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--gcn-warm-up-epoch', type=int, default=0)
    parser.add_argument('--loss-type', type=str, default='CE')

    # args = parser.parse_args()
    #
    # if args.test:
    #     if args.ckpts is None:
    #         raise RuntimeError('--ckpts should not be None when --test is activate')

    return parser

def setup(args):
    args.config = 'config/{}_HP.yaml'.format(args.benchmark)
    args.experiment_path = os.path.join('/experiment_path/', args.benchmark, args.prefix)
    if args.resume:
        resume_path = '/experiment_path/FineDiving/'
        cfg_path = os.path.join(resume_path,'config.yaml')
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return

        print('Resume yaml from %s' % cfg_path)
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        merge_config(config, args)
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        create_experiment_dir(args)
        save_experiment_config(args)

def get_config(args):
    try:
        print('Load config yaml from %s' % args.config)
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    except:
        raise NotImplementedError('%s arch is not supported')
    return config

def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)   

def create_experiment_dir(args):
    try:
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    except:
        pass
    
def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path,'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)