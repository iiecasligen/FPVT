# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step1_source_video2some_frames.py
@time: 2021/9/2 19:31
@describe: 
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def video2frames(video_root, frame_root, extract_num):
    print("# step1: extract some frames from every source video")
    video_name_list = os.listdir(video_root)
    for video_name in tqdm(video_name_list):
        (video_first, video_extension) = os.path.splitext(video_name)
        video_cap = cv2.VideoCapture(os.path.join(video_root, video_name))
        frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        extract_num_list = list(np.linspace(0, frame_num-1, extract_num, endpoint=True, dtype=np.int32))

        for frame_index in extract_num_list:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            cap_success, cap_frame = video_cap.read()
            cv2.imwrite(os.path.join(frame_root, video_first + '__%05d.png' % frame_index), cap_frame)
        video_cap.release()

    print("# step1 Successful: extracted {} * {} frames".format(extract_num, len(video_name_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    arg = parser.add_argument
    arg('--video-root', type=str, default="", help="")
    arg('--frame-root', type=str, default="", help="")
    arg('--extract-num', type=int, default=10, help="")
    args = parser.parse_args()

    video2frames(video_root=args.video_root, frame_root=args.frame_root, extract_num=args.extract_num)
