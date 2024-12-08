import numpy as np
from PIL import Image
import cv2
import math

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_rgb)

def log_transform(image):
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    c = 255 / np.log(1 + np.max(img_array))
    log_image = c * np.log(1 + img_array)
    log_image = np.array(log_image, dtype=np.uint8)
    return Image.fromarray(log_image)

def gamma_correction(image, gamma=1.0):
    img_array = np.array(image, dtype=np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return Image.fromarray(corrected)

def histogram_equalization(image):
    # Convert PIL to OpenCV
    cv2_img = pil_to_cv2(image)
    if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
        # Equalize each channel in HSV or YCrCb
        img_yuv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2YCrCb)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    else:
        equalized = cv2.equalizeHist(cv2_img)
    return cv2_to_pil(equalized)
