import torch
import torch.nn as nn
from model import YoloV1
import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
from train import Compose
from utils import cellboxes_to_boxes, load_checkpoint, get_bboxes
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

import mimetypes


obj_classes = ["bus", "car", "cat", "cow", "dog", "motorbike", "person",  "sheep"]

def filter_and_scale_bboxes(bboxes, prob_thresh = 0.2, image_shape = (448, 448)):
    final_bboxes = []
    W,H = image_shape
    
    sortded_bboxes = sorted(bboxes[0], key = lambda x: x[1], reverse=True)
    
    
    for box in sortded_bboxes:
        if box[1] > prob_thresh:
            final_bboxes.append(
                [box[0],
                 box[2] * W,
                 box[3] * H,
                 box[4] * W,
                 box[5] * H]
            )
    
    return final_bboxes


def draw_bboxes(im: Image, bboxes):
    """
        Draw bounding boxes in the format (xc, yx, width, height)
    """

    draw = ImageDraw.Draw(im)
    for box in bboxes:
        c, xc, yc, h, w = box
        
        xmin = int(xc - w / 2)
        ymin = int(yc - h / 2)
        xmax = int(xc + w / 2)
        ymax = int(yc + h / 2)
        
        object_class = obj_classes[int(c)]
        draw.text((xc,yc), object_class)
        draw.rectangle([(xmin, ymin), (xmax, ymax)])
    
    return im
        
        

def predict(img, model, device='cpu'):
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    im_tensor, _ = transform(img, None)
    
    im_tensor = im_tensor.to(device).unsqueeze(0)
    
    
    pred = model(im_tensor)
    
    bboxes = cellboxes_to_boxes(pred)
    bboxes = filter_and_scale_bboxes(bboxes)
    
    return draw_bboxes(img, bboxes)
    



def load_data_and_predict(in_file, device):
    mimetypes.init()
    mimestart = mimetypes.guess_type(in_file)[0]
    
    if mimestart != None:
        if not ('video' in mimestart or 'image' in mimestart):
            raise Exception("Invalid input file")
    
    else:
        mimestart = 'stream'
    
    
    model_path = "yolo_weights_2"
    model = YoloV1().to(device)
        
    model.load_state_dict(torch.load(model_path)["state_dict"])
        
    model.eval()

    
    if 'image' in mimestart:
        im = Image.open(in_file)
        im = im.resize((448, 448))
        im = predict(im, model, device=device)
        
        plt.imshow(im)
        plt.show()
    
    else:
        print("playing video")
        vcap = cv2.VideoCapture(in_file)
        ret, frame = vcap.read()

        while True:
            
            ret, frame = vcap.read()
            if not ret:
                continue
            
            im = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            im = Image.fromarray(im)
            im = im.resize((448, 448))
            im = predict(im, model, device=device)
            
            
            #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("video", np.asarray(im))
            cv2.waitKey(1)



def main():
    parser = argparse.ArgumentParser(description = 'Make predictiong using trained yolo model')
    
    parser.add_argument('--in_file', help='Input file to tag', type=str, required=True)
    parser.add_argument('--device', help='cuda or cpu, to run the model on.', type=str, required=False, default='cuda')
    
    args = parser.parse_args()
    
    file = args.in_file
    device = args.device
    
    load_data_and_predict(file, device)    

if __name__ == '__main__':
    main()