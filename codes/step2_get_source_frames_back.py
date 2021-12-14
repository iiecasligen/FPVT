# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step2_get_source_frames_back.py
@time: 2021/9/2 20:11
@describe: 
"""
import argparse
import os

import cv2
import numpy as np
import paddlehub as hub
from tqdm import tqdm


def frame2back(frame_root, back_root):
    print("# step2: extract background images from each source frame.")
    frame_name_list = os.listdir(frame_root)
    frame_name_list.sort(reverse=False)
    print("# step2: frame num: {}".format(len(frame_name_list)))

    human_seg = hub.Module(name='deeplabv3p_xception65_humanseg')
    for frame_name in tqdm(frame_name_list):
        frame_path = os.path.join(frame_root, frame_name)
        frame_read = cv2.imread(frame_path)
        frame_mask = human_seg.segmentation(data={'image': [frame_path]})[0]['data']
        frame_mask = cv2.cvtColor(np.array(frame_mask, np.uint8), cv2.COLOR_GRAY2BGR)
        frame_back = (1 - (frame_mask / 255)) * frame_read
        cv2.imwrite(os.path.join(back_root, frame_name), frame_back)
    print("# step2: Successful: extract background images over.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    arg = parser.add_argument
    arg('--frame-root', type=str, default="", help="")
    arg('--back-root', type=str, default="", help="")
    args = parser.parse_args()

    frame2back(frame_root=args.frame_root, back_root=args.back_root)
