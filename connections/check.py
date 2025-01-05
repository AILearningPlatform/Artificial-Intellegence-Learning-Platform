from fastapi import FastAPI, Request, File, UploadFile, Form, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from starlette.requests import Request
from matplotlib import patches
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet50, ResNet50_Weights, VGG16_Weights, vgg16, mobilenet_v2
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

MODEL_FOLDER = "connections/models_or_datasets/"
os.makedirs(MODEL_FOLDER, exist_ok=True)

def save_full_model(model, model_path):
    print(f"Saving model to {model_path}...")
    torch.save(model, model_path)
    print(f"Model saved: {model_path}")
