from fastapi import FastAPI, Request, File, UploadFile, Form, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForCausalLM
from ultralytics import YOLO
from starlette.requests import Request
from matplotlib import patches
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torch_snippets import read as read2, show
import torch.nn.functional as Fn
import torch.nn as nn
from torch_snippets import *
from collections import Counter
from PIL import Image
import torch
import json
import cv2
import numpy as np
import random
import pandas as pd
import httpx
import qrcode
import requests
import asyncio
import os
import shutil
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
    faster_rcnn

)
from torchvision.models import (
    resnet50, ResNet50_Weights, VGG16_Weights, vgg16, mobilenet_v2
)
from torchvision.models.segmentation import deeplabv3_resnet101

MODEL_FOLDER = "connections/models_or_datasets/"
os.makedirs(MODEL_FOLDER, exist_ok=True)

def save_full_model(model, model_path):
    if not os.path.exists(model_path):
        print(f"Saving model to {model_path}...")
        torch.save(model, model_path)
        print(f"Model saved: {model_path}")
    else:
        print(f"Model already exists at {model_path}.")


def ensure_yolo_model(model_name, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        model = YOLO(model_name)
        shutil.move(model_name, model_path)
        print(f"{model_name} downloaded and moved to {model_path}.")
    else:
        print(f"{model_name} already exists at {model_path}.")
    return YOLO(model_path)


print("Checking YOLO models...")
yolo11_model_path = os.path.join(MODEL_FOLDER, "yolo11n.pt")
yolov8_model_path = os.path.join(MODEL_FOLDER, "yolov8s.pt")
yolo11_model = ensure_yolo_model("yolo11n.pt", yolo11_model_path)
yolov8_model = ensure_yolo_model("yolov8s.pt", yolov8_model_path)

print("Checking ResNet50...")
resnet50_model_path = os.path.join(MODEL_FOLDER, "resnet50.pt")
resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
save_full_model(resnet50_model, resnet50_model_path)

print("Checking VGG16...")
vgg16_model_path = os.path.join(MODEL_FOLDER, "vgg16.pt")
vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
save_full_model(vgg16_model, vgg16_model_path)

print("Checking MobileNetV2...")
mobilenet_model_path = os.path.join(MODEL_FOLDER, "mobilenet_v2.pt")
mobilenet_model = mobilenet_v2(weights="IMAGENET1K_V1")
save_full_model(mobilenet_model, mobilenet_model_path)

print("Checking Mask R-CNN...")
mask_rcnn_model_path = os.path.join(MODEL_FOLDER, "mask_rcnn_resnet50_fpn.pt")
mask_rcnn_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
save_full_model(mask_rcnn_model, mask_rcnn_model_path)

print("Checking Faster R-CNN...")
faster_rcnn_model_path = os.path.join(MODEL_FOLDER, "faster_rcnn.pt")
faster_rcnn_model = maskrcnn_resnet50_fpn(weights="DEFAULT")
save_full_model(faster_rcnn_model, faster_rcnn_model_path)

print("Checking deeplabv3_resnet101...")
deeplabv3_model_path = os.path.join(MODEL_FOLDER, "deeplabv3.pt")
deeplabv3_model = deeplabv3_resnet101(weights="DEFAULT")
save_full_model(deeplabv3_model, deeplabv3_model_path)

print("All models are verified and ready!")
