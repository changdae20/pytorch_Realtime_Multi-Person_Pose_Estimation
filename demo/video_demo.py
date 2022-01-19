import os
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

args = {}
args["cfg"] = './experiments/vgg19_368x368_sgd.yaml'
args["weight"] = 'pose_model.pth'
args["opts"] = []
update_config(cfg, args)

batch_size = 400

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model('vgg19')     
model.load_state_dict(torch.load(args["weight"]))
model = torch.nn.DataParallel(model).to(device)
model.eval()
avg_time = 0
total_start = time.time()
for filename in os.listdir("videos/"):
    test_video = "videos/"+filename
    cap = cv2.VideoCapture(test_video) # B,G,R order
    #fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=24
    print("fps : ", fps)
    print("length : ", length)
    width, height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video size : ",width," x ",height)
    out = cv2.VideoWriter("output/"+filename, cv2.VideoWriter_fourcc(*"DIVX"),fps,(width,height))
    # Get results of original image
    count = 0
    while(cap.isOpened()):
        frames = []
        frames_orig = []
        start = time.time()
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            im_croped, _, _ = im_transform.crop_with_factor(
                frame, cfg.DATASET.IMAGE_SIZE, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)
            frames.append(im_croped)
            frames_orig.append(frame)
        if len(frames)==0: break
        frames = np.stack(frames) # Shape : (B,H,W,C)
        #print(np.shape(frames))

        frames = frames.astype(np.float32)
        frames = frames / 256. - 0.5
        frames = frames.transpose((0, 3, 1, 2)).astype(np.float32) # Shape : (B,C,H,W)
        print(np.shape(frames))

        with torch.no_grad():
            frames = torch.from_numpy(frames).float().to(device)
            predicted_outputs, _ = model(frames)
            output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
            heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
            paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)
            for (f,p,h) in zip(frames_orig,paf,heatmap):
                humans = paf_to_pose_cpp(h, p, cfg)
                draw = draw_humans(f, humans)
                out.write(draw)
        
        count += len(frames_orig)
        end = time.time()
        avg_time = ((end-start) + avg_time*(count-1))/count

    cap.release()
    out.release()

total_end = time.time()

print(f"\nTotal Inference Time : {(total_end - total_start):.1f} s")