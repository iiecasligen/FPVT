# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step7_utils_another.py
@time: 2021/9/3 17:47
@describe: 
"""
import os
import pickle

import cv2
import torch
import torch.utils.data.dataset as d_data


def take_last(elem):
    return elem[-1]


def get_flow_pairs(fake_slow_name_list, scence_flow_name_list):
    test_pairs_list = []
    for i in range(len(scence_flow_name_list)):
        test_pairs_precess_one = []
        for scence_flow_name in scence_flow_name_list[i]:
            test_pairs_precess_one.append([fake_slow_name_list[i], scence_flow_name])
        test_pairs_list.append(test_pairs_precess_one)

    return test_pairs_list


def get_cut_image(flow_path, x, y, w, h):
    flow_read = cv2.imread(flow_path)
    cut_flow = flow_read[y:y+h, x:x+w]

    return cut_flow


class ReturnTestFlowCouple2(d_data.Dataset):

    def __init__(self, test_pairs, fake_flow_root, scence_flow_root, fake_rect_pkl, scence_rect_pkl, image_size):
        self.test_pairs = test_pairs
        self.fake_flow_root = fake_flow_root
        self.scence_flow_root = scence_flow_root
        self.image_size = image_size
        with open(fake_rect_pkl, 'rb') as fake_file_pkl:
            self.fake_rect_info = pickle.load(fake_file_pkl)
        fake_file_pkl.close()
        with open(scence_rect_pkl, 'rb') as scence_file_pkl:
            self.scence_rect_info = pickle.load(scence_file_pkl)
        scence_file_pkl.close()

    def __getitem__(self, item):
        pairs_one = self.test_pairs[item]
        get_name_one = pairs_one[0]
        get_name_two = pairs_one[1]
        (x_one, y_one, w_one, h_one) = self.fake_rect_info[get_name_one]['rect']
        (x_two, y_two, w_two, h_two) = self.scence_rect_info[get_name_two]['rect']
        cut_flow_one = get_cut_image(flow_path=os.path.join(self.fake_flow_root, get_name_one),
                                     x=x_one, y=y_one, w=w_one, h=h_one)
        cut_flow_two = get_cut_image(flow_path=os.path.join(self.scence_flow_root, get_name_two),
                                     x=x_two, y=y_two, w=w_two, h=h_two)
        cut_flow_one = cv2.resize(cut_flow_one, (self.image_size, self.image_size))
        cut_flow_two = cv2.resize(cut_flow_two, (self.image_size, self.image_size))
        cut_flow_one = cut_flow_one.reshape(3, self.image_size, self.image_size)
        cut_flow_two = cut_flow_two.reshape(3, self.image_size, self.image_size)
        cut_flow_one = cut_flow_one / 255.
        cut_flow_two = cut_flow_two / 255.
        cut_flow_one = torch.Tensor(cut_flow_one)
        cut_flow_two = torch.Tensor(cut_flow_two)

        return cut_flow_one, cut_flow_two

    def __len__(self):
        len_files = len(self.test_pairs)

        return len_files


def get_angular_dist(f1, f2):
    dist = torch.zeros(len(f1), 1).cuda()
    for i in range(len(f1)):
        dist[i][0] = torch.acos(torch.clamp((torch.sum(torch.mul(f1[i], f2[i])) / torch.clamp(
            torch.norm(f1[i], p=2) * torch.norm(f2[i], p=2), min=1e-8)), min=-1.0, max=1.0))

    return dist

