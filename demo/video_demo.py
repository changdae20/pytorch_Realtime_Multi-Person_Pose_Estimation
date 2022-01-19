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

model = get_model('vgg19')     
model.load_state_dict(torch.load(args["weight"]))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()
avg_time = 0
total_start = time.time()
for filename in os.listdir("videos/"):
    test_video = "videos/"+filename
    cap = cv2.VideoCapture(test_video) # B,G,R order
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #fps=24
    print("fps : ", fps)
    print("length : ", length)
    width, height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video size : ",width," x ",height)
    out = cv2.VideoWriter("output/"+filename, cv2.VideoWriter_fourcc(*"DIVX"),fps,(width,height))
    # Get results of original image
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            start = time.time()
            remain_time = avg_time * (length-count)
            
            sys.stdout.write(f"\r{'■'* (((count+1)*20)//length)}{' '* (20-((count+1)*20)//length)} ({count+1}/{length}) ETA : {int(remain_time)//3600:2d}:{(int(remain_time)%3600)//60:2d}:{int(remain_time)%60:2d}")
            with torch.no_grad():
                paf, heatmap, im_scale = get_outputs(frame, model,  'rtpose')
            #print(im_scale)
            humans = paf_to_pose_cpp(heatmap, paf, cfg)
            draw = draw_humans(frame, humans)
            #cv2.imshow("result",draw)
            #cv2.waitKey(1)
            out.write(draw)
            count += 1
            end = time.time()
            avg_time = ((end-start) + avg_time*(count-1))/count
        else:
            break

    cap.release()
    out.release()

total_end = time.time()

print(f"\nTotal Inference Time : {total_end - total_start:.1f} s")