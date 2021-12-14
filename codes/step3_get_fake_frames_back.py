# -*- coding: utf-8 -*-
"""
@author: LiGen
@file: step3_get_fake_frames_back.py
@time: 2021/9/2 20:21
@describe: 
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import paddlehub as hub
from tqdm import tqdm


def prepare_input_video(video_path, frame_root, rect_root, mask_root, back_root, fore_root):
    print("# step3: extract some frames, background, masks, foreground and rect.")
    print("# step3: from {}".format(video_path))
    (video_first, video_extension) = os.path.splitext(os.path.basename(video_path))
    video_cap = cv2.VideoCapture(video_path)
    frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('# step3: total frame number: {}'.format(frame_num))

    frame_begin_num = int(input("# step3: choose begin num of fake frame: "))
    frame_end_num = int(input("# step3: choose end num of fake frame: "))
    for frame_index in tqdm(range(frame_num)):
        cap_success, cap_frame = video_cap.read()
        if frame_begin_num <= frame_index < frame_end_num:
            cv2.imwrite(os.path.join(frame_root, video_first + '__%05d.png' % frame_index), cap_frame)
    video_cap.release()
    print("# step3: extract {} frames over.".format(frame_end_num-frame_begin_num))

    frame_name_list = os.listdir(frame_root)
    frame_name_list.sort(reverse=False)
    print("# step3: frame num: {}.".format(len(frame_name_list)))

    human_seg = hub.Module(name='deeplabv3p_xception65_humanseg')
    for frame_name in tqdm(frame_name_list):
        frame_path = os.path.join(frame_root, frame_name)
        frame_read = cv2.imread(frame_path)
        frame_mask = human_seg.segmentation(data={'image': [frame_path]})[0]['data']
        frame_mask = cv2.cvtColor(np.array(frame_mask, np.uint8), cv2.COLOR_GRAY2BGR)
        frame_back = (1 - (frame_mask / 255)) * frame_read
        frame_fore = (frame_mask / 255) * frame_read
        cv2.imwrite(os.path.join(back_root, frame_name), frame_back)
        cv2.imwrite(os.path.join(mask_root, frame_name), frame_mask)
        cv2.imwrite(os.path.join(fore_root, frame_name), frame_fore)
    print("# step3: extract background, masks and foreground over.")

    rect_info = {}
    mask_name_list = os.listdir(mask_root)
    mask_name_list.sort(reverse=False)
    for mask_name in tqdm(mask_name_list):
        frame_mask = cv2.imread(os.path.join(mask_root, mask_name))
        mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        mask_ret, mask_binary = cv2.threshold(src=mask_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
        counters, hierarchy = cv2.findContours(image=mask_binary, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        if len(counters) != 1:
            max_id = 0
            max_area = 0
            for i in range(len(counters)):
                x, y, w, h = cv2.boundingRect(counters[i])
                if w * h > max_area:
                    max_area = w * h
                    max_id = i
        else:
            max_id = 0
        max_counter = counters[max_id]
        max_rect = cv2.boundingRect(max_counter)
        rect_info[str(mask_name)] = {'rect': max_rect, 'counter': max_counter}

    rect_name = os.path.basename(video_path).split('.')[0] + '.pkl'
    rect_path = os.path.join(rect_root, rect_name)
    with open(rect_path, 'wb') as file_pkl:
        pickle.dump(rect_info, file_pkl)
    file_pkl.close()

    print("# step3 Successful: save rect pkl file over.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    arg = parser.add_argument
    arg('--video-path', type=str, default="", help="")
    arg('--frame-root', type=str, default="", help="")
    arg('--rect-root', type=str, default="", help="")
    arg('--mask-root', type=str, default="", help="")
    arg('--back-root', type=str, default="", help="")
    arg('--fore-root', type=str, default="", help="")
    args = parser.parse_args()

    prepare_input_video(video_path=args.video_path, frame_root=args.frame_root,
                        rect_root=args.rect_root, mask_root=args.mask_root,
                        back_root=args.back_root, fore_root=args.fore_root)
