# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step4_source_video_map.py
@time: 2021/9/2 20:52
@describe: 
"""
import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader as data_loader
from torch.autograd import Variable
from tqdm import tqdm

from step4_utils_another import get_source_map_pairs, ReturnTestImageCouple2, get_angular_dist, take_last
from step4_utils_net import resnet50


def source_map(fake_back_root, source_back_root, source_video_root, fake_video_path,
               use_gpu_id, image_size, model_path, recompute_num, batch_size, num_workers):
    print("# step4: retrieval source video by scene feature map.")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_id

    print("# step4: load model from {}".format(model_path))
    model = resnet50(pretrained=False, num_classes=1000)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'])

    choose_num = int(input("# step4: choose frame num used for map scene: "))
    test_pairs, image_width, image_height, scence_list = get_source_map_pairs(fake_back_root=fake_back_root,
                                                                              source_back_root=source_back_root,
                                                                              source_video_root=source_video_root,
                                                                              choose_num=choose_num)
    test_data = ReturnTestImageCouple2(fake_back_root=fake_back_root, source_back_root=source_back_root,
                                       test_pairs=test_pairs, image_size=image_size)
    test_loader = data_loader.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, drop_last=False)
    print("# step4: loaded {} = {}/{} pair image.".format(len(test_data), batch_size, len(test_loader)))

    print("# step4: testing.")
    model.eval()
    for i in range(recompute_num):
        for j, (img0, img1) in enumerate(test_loader):
            input0, input1 = Variable(img0.cuda()), Variable(img1.cuda())
            with torch.no_grad():
                output0, output1 = model(input0), model(input1)
            distance = get_angular_dist(output0, output1)
            for data_index in range(len(img0)):
                distance_one = float(distance[data_index][0].cpu().detach().numpy())
                test_pairs[j * batch_size + data_index].append(distance_one)
            print("# step4: [%s] [%d/%d] [%d/%d]" % (datetime.datetime.now().strftime('%H:%M:%S'), i,
                                                     recompute_num, j, len(test_loader)))

    score_list = []
    print("# step4: compare result and get order.")
    for source_video_name in tqdm(scence_list):
        (video_first, video_extension) = os.path.splitext(source_video_name)
        source_one_distance_list = []
        for pair_index in range(len(test_pairs)):
            if test_pairs[pair_index][1][:-11] == video_first:
                for distance_index in range(2, recompute_num + 2):
                    source_one_distance_list.append(test_pairs[pair_index][distance_index])
        scence_one_distance_avg = np.mean(source_one_distance_list)
        score_list.append([source_video_name, scence_one_distance_avg])

    score_list.sort(key=take_last, reverse=False)
    print("# step4: source video score num: {}".format(len(score_list)))
    show_top_num = int(input("# step4: show top score num: "))
    print("# step4: fake video name: {}".format(os.path.basename(fake_video_path)))
    for score_index in range(show_top_num):
        video_name = score_list[score_index][0]
        video_score = score_list[score_index][1]
        print("# step4: video name: %s, video score: %.8f" % (video_name, video_score))

    print("# step4: Successful: get source video.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    arg = parser.add_argument
    arg('--fake_back_root', type=str, default="", help="")
    arg('--source_back_root', type=str, default="", help="")
    arg('--source_video_root', type=str, default="", help="")
    arg('--fake_video_path', type=str, default="", help="")
    arg('--use_gpu_id', type=str, default="0", help="")
    arg('--image_size', type=int, default=256, help="")
    arg('--model_path', type=str, default="", help="")
    arg('--recompute_num', type=int, default=1, help="")
    arg('--batch_size', type=int, default=64, help="")
    arg('--num_workers', type=int, default=5, help="")
    args = parser.parse_args()

    source_map(fake_back_root=args.fake_back_root,
               source_back_root=args.source_back_root,
               source_video_root=args.source_video_root,
               fake_video_path=args.fake_video_path,
               use_gpu_id=args.use_gpu_id,
               image_size=args.image_size,
               model_path=args.model_path,
               recompute_num=args.recompute_num,
               batch_size=args.batch_size,
               num_workers=args.num_workers)
