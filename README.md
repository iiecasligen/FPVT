# FPVT

A Two-level Learning Framework of Fake Portrait Videos Tracing

## Abstract

![FMFCC-V-Dataset](images/sample02.jpg)

Fake Portrait Videos Tracing, which investigates the fake portrait videos and traces source videos from suspected video database at frame level, is a newly proposed challenging task for video forensics. Here, we propose the first solution exploiting two-level learning framework to trace fake portrait videos. Our approach mainly consists of two stages. First, the scene tracing stage aims to retrieve the most similar source video to the input fake portrait video from the suspected video database based on the scene features. Second, the optical flow tracing stage aims to locate the most similar source clip to the input fake portrait video from the retrieved source video based on the motion modes. After that, the source video of the input fake portrait video can be traced at frame level precisely.

## FakeDT Dataset

We released a fake portrait videos tracing dataset, named FakeTD, to support the development of more effective fake portrait videos tracing methods.

The FakeTD dataset can be downloaded from https://pan.baidu.com/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (Password: xxxx).

The structure of FakeDT dataset is:
```
FakeDT
|---FakeDT-1
|---|---fake
|---|---|---FakeTD-1+fake_part0.rar
|---|---|---FakeTD-1+fake_part1.rar
|---|---|---FakeTD-1+fake_part2.rar
|---|---source
|---|---|---FakeTD-1+source_part0.rar
|---FakeDT-2
|---|---fake
|---|---|---FakeTD-2+fake_part0.rar
|---|---source
|---|---|---FakeTD-2+source_part0.rar
```
For the source video of `scene_0000_video_0006.mp4` in `FakeTD-1+source_part0.rar`:
```
0000: index of scene
0006: number of source video with scene 0000
```
For the fake video of `scene_0000_fake_0008.mp4` in `FakeTD-1+fake_part0.rar`:
```
0000: index of scene
0008: number of fake video of scene 0000
```
For the source video of `source_02.mp4` in `FakeTD-2+source_part0.rar`:
```
02: index of source video
```
For the fake video of `source_02_fake_04.mp4` in `FakeTD-2+fake_part0.rar`:
```
02: index of source video
04: number of fake video of source video 02
```

## Demo Guide

### Environment

```
conda create -n trace_env python=3.7
pip install -r trace_env.txt
```

### Testing process

Step 1: extract some frames from source video database
```
python step1_source_video2some_frames.py --video-root ./source_data/ --frame-root ./step1_source_frames/ --extract-num 10
```
Step 2: extract background form source frames extracted in Step 1
```
python step2_get_source_frames_back.py --frame-root ./step1_source_frames/ --back-root ./step2_source_backs/
```
Step 3: extract all frames, masks, backgrounds, foregrounds, rectangle of masks from fake video
```
python step3_get_fake_frames_back.py --video-path ./fake_data/source_01_fake_00.mp4 --frame-root ./step3_fake_frames/ --rect-root ./step3_fake_rect/ --mask-root ./step3_fake_masks/ --back-root ./step3_fake_backs/ --fore-root ./step3_fake_fores/
```







