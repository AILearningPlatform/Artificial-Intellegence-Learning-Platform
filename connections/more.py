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

models = {
    "yolo11": YOLO("static/models_or_datasets/yolo11n.pt")
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
