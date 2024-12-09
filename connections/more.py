from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import transforms
from torch_snippets import *
from torch_snippets import read as read2
import cv2
from ultralytics import YOLO
import random
