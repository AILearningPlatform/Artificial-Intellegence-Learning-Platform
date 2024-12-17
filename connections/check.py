from fastapi import FastAPI, Request, File, UploadFile, Form 
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates 
from ultralytics import YOLO 
from matplotlib import patches   
import matplotlib.pyplot as plt  
import torch 
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights  
from torchvision.models import resnet50, ResNet50_Weights 
from torchvision.transforms import transforms as T 
from torch_snippets import read as read2 
import torch.nn as nn 
from torch_snippets import * 
from PIL import Image 
from collections import Counter 
import cv2 
import numpy as np 
import random 
import pandas as pd
