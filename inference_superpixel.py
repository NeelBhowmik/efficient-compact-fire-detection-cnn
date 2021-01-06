################################################################################

# Example : perform live fire detection in image/video/webcam using superpixel localisation
# and the superpixel trained version of the NasNet-A-OnFire, ShuffleNetV2-OnFire CNN models.

# Copyright (c) 2020/21 - William Thompson / Neelanjan Bhowmik / Toby Breckon, Durham University, UK

# License : https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn/blob/main/LICENSE

################################################################################

import cv2
import os
import sys
import math
from PIL import Image
import argparse
import time
import numpy as np

################################################################################

import torch
import torchvision.transforms as transforms
from models import shufflenetv2
from models import nasnet_mobile_onfire

################################################################################

def data_transform(model):
    # transforms needed for shufflenetonfire
    if model == 'shufflenetonfire':
        np_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    # transforms needed for nasnetonfire
    if model == 'nasnetonfire':
        np_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    return np_transforms

################################################################################

def proc_sp(small_frame, np_transforms):
    # small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
    small_frame = Image.fromarray(small_frame)
    small_frame = np_transforms(small_frame).float()
    small_frame = small_frame.unsqueeze(0)
    small_frame =  small_frame.to(device)
    
    return small_frame 

################################################################################

def pil_crop(img):
    sp_pil = Image.fromarray(img)
    imageBox = sp_pil.getbbox()
    sp_crop = sp_pil.crop(imageBox)
    np_sp_crop = np.array(sp_crop)  
    # sp_crop_cv = cv2.cvtColor(np_sp_crop, cv2.COLOR_RGB2BGR)
    return np_sp_crop 

################################################################################

def run_model_img(args, frame, model):
    output = model(frame)[0]
    pred = torch.round(torch.sigmoid(output))
    return pred

################################################################################

def draw_pred(args, frame, contours, prediction):
    # height, width, _ = frame.shape
    if prediction == 1:
        cv2.drawContours(frame, contours, -1, (0,0,255), 1)
    else:
        cv2.drawContours(frame, contours, -1, (0,255,0), 1)    
    return frame

################################################################################

def process_sp(args, small_frame, np_transforms, model):
    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)
    segments = slic.getLabels()
    
    for (i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(small_frame.shape[:2], dtype='uint8')
        mask[segments == segVal] = 255

        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contour_list.append(contours)
        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)
        superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)
        
        #PIL centre crop and data transformation
        # superpixel = pil_crop(superpixel)
        superpixel = proc_sp(superpixel, np_transforms)

        # Model prediction
        prediction = run_model_img(args, superpixel, model)
       
        # Draw prediction on superpixel 
        draw_pred(args, small_frame, contours, prediction)

################################################################################
# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Path to image file or image directory")
parser.add_argument("--video", help="Path to video file or video directory")
parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
parser.add_argument("--trt", action="store_true", help="Model run on TensorRT")
parser.add_argument("--model", default='shufflenetonfire', help="Select the model {shufflenetonfire, nasnetonfire}")
parser.add_argument("--cpu", action="store_true", help="If selected will run on CPU")
parser.add_argument(
    "--output", 
    help="A directory to save output visualizations."
    "If not given , will show output in an OpenCV window."
)
args = parser.parse_args()
print(f'\n{args}')

WINDOW_NAME = 'Detection'
# uses cuda if available
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.cpu and args.trt:
    print(f'\n>>>>TensorRT runs only on gpu. Exit.')
    exit()

print('\n\nBegin {fire, no-fire} superpixel localisation :')

# model load
if args.model == "shufflenetonfire":
    model = shufflenetv2.shufflenet_v2_x0_5(pretrained=False, layers=[4, 8, 4], output_channels=[24, 48, 96, 192, 64], num_classes=1)
    model.load_state_dict(torch.load('./weights/shufflenet_sp.pt', map_location=device))
elif args.model == "nasnetonfire":
    model = nasnet_mobile_onfire.nasnetamobile(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load('./weights/nasnet_sp.pt', map_location=device))
else:
    print('Invalid Model.')
    exit()

np_transforms = data_transform(args.model)

print(f'|__Model loading: {args.model}')

model.eval()
model.to(device)

# TensorRT conversion
if args.trt:
    from torch2trt import TRTModule
    from torch2trt import torch2trt
    data = torch.randn((1, 3, 224, 224)).float().to(device)
    model_trt = torch2trt(model, [data], int8_mode=True)
    model_trt.to(device)
    print(f'\t|__TensorRT activated.')

# load and process input image
if args.image:
    if os.path.isdir(args.image):
        fps = []
        for im in os.listdir(args.image):
            print('\t|____Image processing: ', im)
            
            frame = cv2.imread(f'{args.image}/{im}')
            small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
            
            # Prediction on superpixel
            if args.trt:
                process_sp(args, small_frame, np_transforms, model_trt)
            else:
                process_sp(args, small_frame, np_transforms, model)
            
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                cv2.imwrite(f'{args.output}/{im}', small_frame)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, small_frame)
                cv2.waitKey(0)

    else:
        print('\t|____Image processing: ', args.image)
        
        frame = cv2.imread(f'{args.image}')
        small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
                
        # Prediction on superpixel
        if args.trt:
            process_sp(args, small_frame, np_transforms, model_trt)
        else:
            process_sp(args, small_frame, np_transforms, model)
        
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            cv2.imwrite(f'{args.output}/{args.image.split("/")[-1]}', small_frame)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, small_frame)
            cv2.waitKey(0)

# load and process input video
if args.video:
    if os.path.isdir(args.video):
        for vid in os.listdir(args.video):
            print('\t|____Video processing: ', vid)
            video = cv2.VideoCapture(f'{args.video}/{vid}')
            keepProcessing = True

            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            img_array = []

            while (keepProcessing):
                ret, frame = video.read()

                if not ret:
                    print("\t\t... end of video.")
                    break

                small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
                                            
                # Prediction on superpixel
                if args.trt:
                    process_sp(args, small_frame, np_transforms, model_trt)
                else:
                    process_sp(args, small_frame, np_transforms, model)
                img_array.append(small_frame)

                if args.output:
                    os.makedirs(args.output, exist_ok=True)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, small_frame)
                    cv2.waitKey(int(fps))
            
            if args.output:
                out = cv2.VideoWriter(
                    filename=f'{args.output}/{vid}', 
                    fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
                    fps=float(fps), 
                    frameSize=(width, height),
                    isColor=True,
                )
                
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
            
                           
    else:
        print('\t|____Video processing: ', args.video)
        video = cv2.VideoCapture(f'{args.video}')
        keepProcessing = True

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        img_array = []

        while (keepProcessing):
            ret, frame = video.read()

            if not ret:
                print(f"\t\t... end of video.")
                break

            small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
                                            
            # Prediction on superpixel
            if args.trt:
                process_sp(args, small_frame, np_transforms, model_trt)
            else:
                process_sp(args, small_frame, np_transforms, model)
            img_array.append(small_frame)

            if args.output:
                os.makedirs(args.output, exist_ok=True)
                
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, small_frame)
                cv2.waitKey(int(fps))

        if args.output:
            out = cv2.VideoWriter(
                filename=f'{args.output}/{args.video.split("/")[-1]}', 
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
                fps=float(fps), 
                frameSize=(width, height),
                isColor=True,
            )
            
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()
# load and process input webcam
if args.webcam:
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        success, frame = cam.read()
        if success:
            small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
            
            # Prediction on superpixel
            if args.trt:
                process_sp(args, small_frame, np_transforms, model_trt)
            else:
                process_sp(args, small_frame, np_transforms, model)
            
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, small_frame)
            if cv2.waitKey(1) == 27:
                exit()
            
print('\n[Done]\n')