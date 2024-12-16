from fastapi import FastAPI, Request, File, UploadFile, Form #type: ignore
from fastapi.responses import HTMLResponse, RedirectResponse #type: ignore
from fastapi.staticfiles import StaticFiles #type: ignore
from fastapi.templating import Jinja2Templates #type: ignore
from PIL import Image #type: ignore
import torch #type: ignore
from torchvision import transforms #type: ignore
from torch_snippets import * #type: ignore
from torch_snippets import read as read2 #type: ignore
import cv2 #type: ignore
from ultralytics import YOLO #type: ignore
import random 
import numpy as np #type: ignore
from collections import Counter

models = {
    "yolo11": YOLO("static/models_or_datasets/yolo11n.pt"),
    "yolov8": YOLO("static/models_or_datasets/yolov8n.pt")
}

class Models:
    @staticmethod
    def Mask_R_CNN_Instance_Segmentation(image_path):
        return [f"Image: {image_path[18:]} \nPredicted: Gwapo", image_path]

    @staticmethod
    def ResNet_50_Image_Classification(image_path):
        return [f"Image: {image_path[18:]} Predicted: Gwapo", image_path]

    @staticmethod
    def CycleGAN_Image_to_Image_Translation(image_path):
        return [f"Image: {image_path[18:]} Predicted: Gwapo", image_path]

    @staticmethod
    def YOLOv11_Object_Detection(image_path):
        path = image_path
        img = read(path)
        pred = models['yolo11'](img)

        boxes = pred[0].boxes
        labels = boxes.cls  

        all_preds = [(pred[0].names[labels[i].item()], boxes.xyxy[i].cpu().numpy().tolist()) for i in range(len(labels))]

        class_counts = Counter([name for name, _ in all_preds])

        if class_counts:
            result = "Predictions: " + ", ".join([f"{count} {name}s" for name, count in class_counts.items()])
        else:
            result = "No predictions."

        show(img, bbs=boxes.xyxy.cpu().numpy(), title=result)

        return [f"Image: {image_path[18:]} {result}", image_path]



