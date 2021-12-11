# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step6_get_flow.py
@time: 2021/9/3 14:22
@describe: 
"""
import argparse
import os

import cv2
import torch
from tqdm import tqdm

from .step6_utils_another import flow_to_image, InputPadder, load_image, get_image_list
from .step6_utils_net import RAFT


def get_flow(fake_frame_root, source_frame_root, fake_mask_root, source_mask_root,
             fake_flow_root, source_flow_root, model_path, use_gpu_id):
    print("# step6: get all flow from fake frames and scence frames")
    fake_frame_pairs_list = get_image_list(image_root=fake_frame_root)
    source_frame_pairs_list = get_image_list(image_root=source_frame_root)
    print("# step6: fake frame pair is: {}".format(len(fake_frame_pairs_list)))
    print("# step6: scence frame pair is: {}".format(len(source_frame_pairs_list)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_id

    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--small', action='store_true', help='use small model')
    parser2.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser2.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    parser_argument = parser2.parse_args()

    print("# step6: load model from {}".format(model_path))
    model = torch.nn.DataParallel(RAFT(parser_argument))
    model.load_state_dict(torch.load(model_path))
    model = model.module
    model.to('cuda:0')
    model.eval()

    print("# step6: get fake frame flow")
    for (img_path0, img_path1) in tqdm(fake_frame_pairs_list):
        img0 = load_image(img_path=img_path0)[None].to('cuda:0')
        img1 = load_image(img_path=img_path1)[None].to('cuda:0')
        padder = InputPadder(img0.shape)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad():
            flow_low, flow_up = model(img0, img1, iters=32, test_mode=True)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_up = flow_to_image(flow_up)
        mask_read = cv2.imread(os.path.join(fake_mask_root, os.path.basename(img_path0)))
        flow_fore = (mask_read / 255) * flow_up + 255 - mask_read
        cv2.imwrite(os.path.join(fake_flow_root, os.path.basename(img_path0)), flow_fore)

    print("# step6: get scence frame flow")
    for (img_path0, img_path1) in tqdm(source_frame_pairs_list):
        img0 = load_image(img_path=img_path0)[None].to('cuda:0')
        img1 = load_image(img_path=img_path1)[None].to('cuda:0')
        padder = InputPadder(img0.shape)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad():
            flow_low, flow_up = model(img0, img1, iters=32, test_mode=True)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_up = flow_to_image(flow_up)
        mask_read = cv2.imread(os.path.join(source_mask_root, os.path.basename(img_path0)))
        flow_fore = ((mask_read / 255) * flow_up) + ((1 - (mask_read / 255)) * 255)
        cv2.imwrite(os.path.join(source_flow_root, os.path.basename(img_path0)), flow_fore)

    print("# step6 Successful: get flow over")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    arg = parser.add_argument
    arg('--fake_frame_root', type=str, default="", help="")
    arg('--fake_mask_root', type=str, default="", help="")
    arg('--fake_flow_root', type=str, default="", help="")
    arg('--source_frame_root', type=str, default="", help="")
    arg('--source_mask_root', type=str, default="", help="")
    arg('--source_flow_root', type=str, default="", help="")
    arg('--model_path', type=str, default="", help="")
    arg('--use_gpu_id', type=str, default="0", help="")
    args = parser.parse_args()

    get_flow(fake_frame_root=args.fake_frame_root,
             fake_mask_root=args.fake_mask_root,
             fake_flow_root=args.fake_flow_root,
             source_frame_root=args.source_frame_root,
             source_mask_root=args.source_mask_root,
             source_flow_root=args.source_flow_root,
             model_path=args.model_path,
             use_gpu_id=args.use_gpu_id)
