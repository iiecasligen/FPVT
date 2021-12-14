# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step7_flow_map.py
@time: 2021/9/3 15:50
@describe: 
"""
import argparse
import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data.dataloader as data_loader
from torch.autograd import Variable

from step4_utils_net import resnet50
from step7_utils_another import get_flow_pairs, ReturnTestFlowCouple2, get_angular_dist, take_last


def flow_map(fake_flow_root, source_flow_root, use_gpu_id, model_path,
             fake_rect_pkl, source_rect_pkl, image_size, batch_size, num_workers):
    print("# step7: get source video flow index by flow map")
    fake_flow_name_list = os.listdir(fake_flow_root)
    fake_flow_name_list.sort(reverse=False)
    print("# step7: fake flow number: {}".format(len(fake_flow_name_list)))
    frame_index_begin = 0
    frame_index_end = len(fake_flow_name_list) + 1
    flow_index_begin = frame_index_begin
    flow_index_end = frame_index_end - 1
    choose_num = int(input("# step7: choose sample num of map flow: "))
    get_fake_flow_index_list = list(np.linspace(flow_index_begin, flow_index_end - 1, choose_num,
                                                endpoint=True, dtype=np.int32))
    get_fake_flow_index_list2 = list(np.linspace(0, flow_index_end - 1 - flow_index_begin, choose_num,
                                                 endpoint=True, dtype=np.int32))
    get_fake_flow_name_list = []
    for flow_index in get_fake_flow_index_list:
        get_fake_flow_name_list.append(fake_flow_name_list[flow_index])
    get_fake_flow_name_list.sort(reverse=False)

    source_flow_name_list = os.listdir(source_flow_root)
    source_flow_name_list.sort(reverse=False)
    source_flow_name_list_num = len(source_flow_name_list)
    print("# step7: source flow number: {}".format(source_flow_name_list_num))

    get_source_flow_name_list = []
    get_source_flow_name_list_len = source_flow_name_list_num - (flow_index_end - flow_index_begin) + 1
    for compute_index in get_fake_flow_index_list2:
        begin_num = compute_index
        end_num = get_source_flow_name_list_len + compute_index
        source_flow_name_list_one = source_flow_name_list[begin_num:end_num]
        get_source_flow_name_list.append(source_flow_name_list_one)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_id

    print("# step7: load model from {}".format(model_path))
    model = resnet50(pretrained=False, num_classes=1000)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'])

    test_pairs_list = get_flow_pairs(fake_slow_name_list=get_fake_flow_name_list,
                                     scence_flow_name_list=get_source_flow_name_list)
    model.eval()
    score_list = []
    print("# step7: testing")
    for choose_num_index in range(choose_num):
        score_list_one = []
        test_pairs_list_one = test_pairs_list[choose_num_index]
        test_data_one = ReturnTestFlowCouple2(test_pairs=test_pairs_list_one, fake_flow_root=fake_flow_root,
                                              scence_flow_root=source_flow_root, fake_rect_pkl=fake_rect_pkl,
                                              scence_rect_pkl=source_rect_pkl, image_size=image_size)
        test_loader_one = data_loader.DataLoader(dataset=test_data_one, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, drop_last=False)
        print("# step7: loaded {} = {}/{} pair image".format(len(test_data_one), batch_size, len(test_loader_one)))

        for i, (flow0, flow1) in enumerate(test_loader_one):
            input0, input1 = Variable(flow0.cuda()), Variable(flow1.cuda())
            with torch.no_grad():
                output0, output1 = model(input0), model(input1)
            distance = get_angular_dist(output0, output1)
            for data_index in range(len(flow0)):
                distance_one = float(distance[data_index][0].cpu().detach().numpy())
                score_list_one.append(distance_one)
            print("# step7: [%s] [%d/%d] [%d/%d]" % (datetime.datetime.now().strftime('%H:%M:%S'), choose_num_index,
                                                     choose_num, i, len(test_loader_one)))
        score_list.append(score_list_one)

    map_info = []
    for index_i in range(get_source_flow_name_list_len):
        score_one = 0.0
        for index_j in range(choose_num):
            score_one += score_list[index_j][index_i]
        map_info.append([index_i, score_one / choose_num])

    map_info.sort(key=take_last, reverse=False)
    print("# step7: flow score num: {}".format(len(map_info)))
    show_top_num = int(input("# step7: show top score num: "))
    for score_index in range(show_top_num):
        flow_index_location_begin = map_info[score_index][0]
        flow_index_location_end = flow_index_location_begin + (flow_index_end - flow_index_begin)
        frame_index_location_end = flow_index_location_end + 1
        video_score = map_info[score_index][1]
        print("# step7: flow index: [%d, %d], frame index: [%d, %d], score: %.8f" % (flow_index_location_begin,
                                                                                     flow_index_location_end,
                                                                                     flow_index_location_begin,
                                                                                     frame_index_location_end,
                                                                                     video_score))

    print("# step7: Successful: get location")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    arg = parser.add_argument
    arg('--fake_rect_pkl', type=str, default="", help="")
    arg('--fake_flow_root', type=str, default="", help="")
    arg('--source_rect_pkl', type=str, default="", help="")
    arg('--source_flow_root', type=str, default="", help="")
    arg('--model_path', type=str, default="", help="")
    arg('--use_gpu_id', type=str, default="0", help="")
    arg('--image_size', type=int, default=256, help="")
    arg('--batch_size', type=int, default=64, help="")
    arg('--num_workers', type=int, default=5, help="")
    args = parser.parse_args()

    flow_map(fake_flow_root=args.fake_flow_root,
             source_flow_root=args.source_flow_root,
             use_gpu_id=args.use_gpu_id,
             model_path=args.model_path,
             fake_rect_pkl=args.fake_rect_pkl,
             source_rect_pkl=args.source_rect_pkl,
             image_size=args.image_size,
             batch_size=args.batch_size,
             num_workers=args.num_workers)
