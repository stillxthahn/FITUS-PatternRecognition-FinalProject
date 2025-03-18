import torch
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN
import cv2
import yaml
import datetime
from TDDFA import TDDFA
from utils.render import render
from utils.uv import uv_tex
from utils.serialization import ser_to_obj
from utils.functions import draw_landmarks

IMG_PATH = "images/img1.jpg"

# Determine if an nvidia GPU is available
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,
    keep_all=True
)

img = Image.open(IMG_PATH)
img2 = cv2.imread(IMG_PATH)
img_np = np.array(img)

# Step 1: Get boxes and probability
boxes, probs, points = mtcnn.detect(img, landmarks=True)

# # Step 2: TDDFA
cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)

name_save= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_save =os.path.join("output", str(name_save))
os.makedirs(path_save, exist_ok = True)

param_lst, roi_box_lst = tddfa(img_np, boxes)

# Landmark visualization
ver_landmark = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
path_imglandmark = path_save + '/data_landmarks.jpg'
draw_landmarks(img2, ver_landmark, show_flag=False, dense_flag=False, wfp=path_imglandmark)

# Dense face reconstruction
ver_3d = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

# Obj 
wfp_obj = path_save + '/data3d.obj'
ser_to_obj(img2, ver_3d, tddfa.tri, height=img2.shape[0], wfp=wfp_obj)

# Texture face
path_textureface = path_save + '/data_textureface.jpg'
uv_tex(img2, ver_3d, tddfa.tri, show_flag=False, wfp=path_textureface)

# 3D Render
path_3drender = path_save + '/data_3drender.jpg'
render(img2, ver_3d, tddfa.tri, alpha=0.6, show_flag=False, wfp=path_3drender)
