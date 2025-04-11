import torch
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN
import dlib
import cv2
import yaml
from TDDFA import TDDFA
from utils.render import render
from utils.uv import uv_tex
from utils.serialization import ser_to_obj
from utils.functions import draw_landmarks

# Path to the folder containing images
IMG_FOLDER = "DATA/"
NAME_SAVE = ""
PATH_SAVE = os.path.join("output", str(NAME_SAVE))
os.makedirs(PATH_SAVE, exist_ok=True)

# Files to save images that cannot be detected
FACENET_FAIL_FILE = os.path.join(PATH_SAVE, "facenet_failures.txt")
# DLIB_FAIL_FILE = os.path.join(PATH_SAVE, "dlib_failures.txt")

# Determine if an NVIDIA GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize Facenet
mtcnn = MTCNN(
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,
    keep_all=True
)

# Initialize Dlib
# dlib_detector = dlib.get_frontal_face_detector()

# Load TDDFA configuration
cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)

# Open files to record failures
with open(FACENET_FAIL_FILE, "w") as facenet_fail:
    # Process all JPG files in the folder
    for filename in os.listdir(IMG_FOLDER):
        if filename.endswith(".jpg"):
            img_path = os.path.join(IMG_FOLDER, filename)
            print(f"Processing {img_path}...")

            try:
                # Load image
                img = Image.open(img_path)
                img2 = cv2.imread(img_path)
                # img_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                img_np = np.array(img)

                # Step 1: Get boxes and probability
                # Facenet
                check = True
                fn_boxes, fn_probs, fn_points = mtcnn.detect(img, landmarks=True)
                if fn_boxes is None:
                    print(f"No face detected in {img_path} with Facenet")
                    facenet_fail.write(f"{img_path}\n")
                    check = False

                # Step 2: TDDFA
                fn_param_lst, fn_roi_box_lst = tddfa(img2, fn_boxes)

                # Landmark visualization
                ver_landmark = tddfa.recon_vers(fn_param_lst, fn_roi_box_lst, dense_flag=False)
                path_imglandmark = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_facenet_landmarks.jpg")
                draw_landmarks(img2, ver_landmark, show_flag=False, dense_flag=False, wfp=path_imglandmark)


                # Dense face reconstruction
                fn_ver_3d = tddfa.recon_vers(fn_param_lst, fn_roi_box_lst, dense_flag=True)

                
                fb_textureface = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_facenet_texture.jpg")
                uv_tex(img2, fn_ver_3d, tddfa.tri, show_flag=False, wfp=fb_textureface)

                # Obj
                fn_wfp_obj = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_facenet.obj")
                ser_to_obj(img2, fn_ver_3d, tddfa.tri, height=img2.shape[0], wfp=fn_wfp_obj)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Processing completed.")