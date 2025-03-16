import os
import pickle
import sys
import torch
import yaml

from tools import train_net
from utils import parser
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import ssl

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    args_parser = parser.get_args()
    args = args_parser.parse_args()
    if args.GCN_config is not None:
        with open(args.GCN_config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        args_parser.set_defaults(**default_arg)
    args = args_parser.parse_args()
    parser.setup(args)
    print(args)
    train_net(args)


if __name__ == '__main__':
    main()

