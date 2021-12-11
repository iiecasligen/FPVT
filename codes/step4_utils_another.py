# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step4_utils_another.py
@time: 2021/9/3 8:49
@describe: 
"""
import os
import random

import cv2
import torch
import torch.utils.data.dataset as d_data


def take_last(elem):
    return elem[-1]


def get_source_map_pairs(fake_back_root, source_back_root, source_video_root, choose_num):
    fake_back_name_list = os.listdir(fake_back_root)
    fake_back_name_list.sort(reverse=False)
    print("# step4: fake frame number: {}".format(len(fake_back_name_list)))
    frame_index_begin = int(input("# step4: input sample frame number begin (include): "))
    frame_index_end = int(input("# step4: input sample frame number end (not include): "))
    fake_back_name_list = fake_back_name_list[frame_index_begin:frame_index_end]
    get_fake_back_name_list = random.sample(population=fake_back_name_list, k=choose_num)
    image_sample_read = cv2.imread(os.path.join(fake_back_root, get_fake_back_name_list[0]))
    image_width = image_sample_read.shape[1]
    image_height = image_sample_read.shape[0]
    source_back_name_list = os.listdir(source_back_root)
    source_video_name_list = os.listdir(source_video_root)
    source_video_name_list.sort(reverse=False)
    get_source_back_name_list = []
    for source_video_name in source_video_name_list:
        (video_first, video_extension) = os.path.splitext(source_video_name)
        source_back_name_list_one = []
        for source_back_name_one in source_back_name_list:
            if video_first in source_back_name_one:
                source_back_name_list_one.append(source_back_name_one)
        get_source_back_name_list_one = random.sample(population=source_back_name_list_one, k=choose_num)
        get_source_back_name_list += get_source_back_name_list_one

    test_pairs = []
    for fake_sample in get_fake_back_name_list:
        for source_sample in get_source_back_name_list:
            test_pairs.append([fake_sample, source_sample])

    return test_pairs, image_width, image_height, source_video_name_list


class ReturnTestImageCouple2(d_data.Dataset):

    def __init__(self, fake_back_root, source_back_root, test_pairs, image_width, image_height, sample_cut, image_size):
        self.fake_back_root = fake_back_root
        self.source_back_root = source_back_root
        self.test_pairs = test_pairs
        self.image_width = image_width
        self.image_height = image_height
        self.sample_cut = sample_cut
        self.image_size = image_size

    def __getitem__(self, item):
        pairs_one = self.test_pairs[item]
        img_read_one = cv2.imread(os.path.join(self.fake_back_root, pairs_one[0]))
        img_read_two = cv2.imread(os.path.join(self.source_back_root, pairs_one[1]))
        if img_read_two.shape[1] != self.image_width or img_read_two.shape[0] != self.image_height:
            img_read_two = cv2.resize(img_read_two, (self.image_width, self.image_height))

        cut_x = random.randint(0, self.image_width - self.sample_cut)
        cut_y = random.randint(0, self.image_height - self.sample_cut)
        img_read_one = img_read_one[cut_y:(cut_y + self.sample_cut), cut_x:(cut_x + self.sample_cut)]
        img_read_two = img_read_two[cut_y:(cut_y + self.sample_cut), cut_x:(cut_x + self.sample_cut)]
        if self.sample_cut != self.image_size:
            img_read_one = cv2.resize(img_read_one, (self.image_size, self.image_size))
            img_read_two = cv2.resize(img_read_two, (self.image_size, self.image_size))

        img_read_one = img_read_one.reshape(3, self.image_size, self.image_size)
        img_read_two = img_read_two.reshape(3, self.image_size, self.image_size)
        img_read_one = img_read_one / 255.
        img_read_two = img_read_two / 255.
        img_read_one = torch.Tensor(img_read_one)
        img_read_two = torch.Tensor(img_read_two)

        return img_read_one, img_read_two

    def __len__(self):
        len_files = len(self.test_pairs)

        return len_files


def get_angular_dist(f1, f2):
    dist = torch.zeros(len(f1), 1).cuda()
    for i in range(len(f1)):
        dist[i][0] = torch.acos(torch.clamp((torch.sum(torch.mul(f1[i], f2[i])) / torch.clamp(
            torch.norm(f1[i], p=2) * torch.norm(f2[i], p=2), min=1e-8)), min=-1.0, max=1.0))

    return dist
