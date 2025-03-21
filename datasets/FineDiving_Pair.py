import json

import torch
import numpy as np
import os
import pickle
import random
import glob
from os.path import join
from PIL import Image

from datasets import tools


class FineDiving_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        # self.random_choosing = args.random_choosing
        self.action_number_choosing = args.action_number_choosing
        self.DD_choosing = args.DD_choosing
        self.length = args.frame_length
        self.voter_number = args.voter_number
        # self.data_set = args.data_set
        self.Pose_flag = args.Pose_flag
        # file path
        self.data_root = args.data_root
        self.data_anno = self.read_pickle(args.label_path)
        self.action_number = 1
        self.collect_action_number()


        self.Heat_flag = args.Heat_flag
        with open(args.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)

        # pose path
        if args.Pose_Dataset == None:
            self.flag_direct_pose_heatmap_numpy = 0
            self.flag_direct_heatmap_numpy = 0
        else:
            self.flag_direct_pose_heatmap_numpy = 1
            self.flag_direct_heatmap_numpy = 0
            self.pose_root = args.Pose_Anno_path
            if args.Heat_flag:
                self.flag_direct_heatmap_numpy = 1
                self.heat_root = args.Heat_Anno_path


        self.action_number_dict = {}
        self.difficulties_dict = {}
        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def collect_action_number(self):
        action_mapping = {}
        for key in self.data_anno:
            current_action = self.data_anno[key][0]
            if current_action not in action_mapping:
                action_mapping[current_action] = self.action_number
                self.action_number += 1
            self.data_anno[key].append(action_mapping[current_action])


    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.data_anno.get(item)[0]
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
        if self.subset == 'test':
            for item in self.test_dataset_list:
                dive_number = self.data_anno.get(item)[0]
                if self.action_number_dict_test.get(dive_number) is None:
                    self.action_number_dict_test[dive_number] = []
                self.action_number_dict_test[dive_number].append(item)

    def check_exemplar_dict(self):
        if self.subset == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key
        if self.subset == 'test':
            for key in sorted(list(self.action_number_dict_test.keys())):
                file_list = self.action_number_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key



    def load_video(self, video_file_name):
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))
        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])

        frame_list = np.linspace(start_frame, end_frame, self.length).astype(np.int)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
        frames_labels = [self.data_anno.get(video_file_name)[4][i] for i in image_frame_idx]
        frames_catogeries = list(set(frames_labels))
        frames_catogeries.sort(key=frames_labels.index)
        transitions = [frames_labels.index(c) for c in frames_catogeries]
        return self.transforms(video), np.array([transitions[1] - 1, transitions[-1] - 1]), np.array(frames_labels)

    def load_pose_posemaps(self, video_file_name):
        if self.flag_direct_pose_heatmap_numpy:
            match_dir = os.path.join(self.pose_root, video_file_name[0])
            vid = str(video_file_name[1])
            pred_array = 'pred_comp_' + vid + '.npz'
            pose_pred = np.load(os.path.join(match_dir, pred_array))['pred']
            pose_pred = pose_pred.reshape((96, 16, 2, 1)).transpose(2, 0, 1, 3)
            pred_info = torch.from_numpy(pose_pred).float()
        else:
            pred_info = torch.zeros(16, self.length, 64, 64).float()
        return pred_info

    def load_pose_heatmaps(self, video_file_name):
        if self.flag_direct_heatmap_numpy:
            match_dir = os.path.join(self.heat_root, video_file_name[0])
            vid = str(video_file_name[1])
            pred_array = 'heatmap_comp_' + vid + '.npz'
            pose_heatmaps = np.load(os.path.join(match_dir, pred_array))['heat']
            heatmap_info = torch.from_numpy(pose_heatmaps).permute(1, 0, 2, 3).float()
        else:
            heatmap_info = torch.zeros(16, self.length, 64, 64).float()
        return heatmap_info



    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample_1  = self.dataset[index]
        data = {}
        data['video'], data['transits'], data['frame_labels'] = self.load_video(sample_1)
        data['number'] = self.data_anno.get(sample_1)[5]
        data['final_score'] = self.data_anno.get(sample_1)[1]
        data['difficulty'] = self.data_anno.get(sample_1)[2]
        data['completeness'] = (data['final_score'] / data['difficulty'])

        if self.Pose_flag:
            data['pred_info'] = self.load_pose_posemaps(sample_1)
        # choose a exemplar
        if self.subset == 'train':
            # train phrase
            if self.action_number_choosing == True:
                file_list = self.action_number_dict[self.data_anno[sample_1][0]].copy()
            elif self.DD_choosing == True:
                file_list = self.difficulties_dict[self.data_anno[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            target['video'], target['transits'], target['frame_labels'] = self.load_video(sample_2)
            target['number'] = self.data_anno.get(sample_2)[5]
            target['final_score'] = self.data_anno.get(sample_2)[1]
            target['difficulty'] = self.data_anno.get(sample_2)[2]
            target['completeness'] = (target['final_score'] / target['difficulty'])
            # heatmap
            if self.Pose_flag:
                target['pred_info'] = self.load_pose_posemaps(sample_2)
            return data, target
        else:
            # test phrase
            if self.action_number_choosing:
                train_file_list = self.action_number_dict[self.data_anno[sample_1][0]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            elif self.DD_choosing:
                train_file_list = self.difficulties_dict[self.data_anno[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'], tmp['transits'], tmp['frame_labels'] = self.load_video(item)
                tmp['number'] = self.data_anno.get(item)[5]
                tmp['final_score'] = self.data_anno.get(item)[1]
                tmp['difficulty'] = self.data_anno.get(item)[2]
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])
                # heatmap
                if self.Pose_flag:
                    tmp['pred_info'] = self.load_pose_posemaps(item)
                target_list.append(tmp)
            return data, target_list

    def __len__(self):
        return len(self.dataset)
