import os #directories, join path
import glob
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import re


def read_image(content: bytes) -> np.ndarray:
    """
    Image bytes to OpenCV image

    :param content: Image bytes
    :returns OpenCV image
    :raises TypeError: If content is not bytes
    :raises ValueError: If content does not represent an image
    """
    if not isinstance(content, bytes):
        raise TypeError(f"Expected 'content' to be bytes, received: {type(content)}")
    image = cv.imdecode(np.frombuffer(content, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Expected 'content' to be image bytes")
    return image


def preprocessing_data(img_dir):
  # Reading Image
#   img = cv2.imread(img_dir)
  img = read_image(img_dir)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR(opencv format) to RGB format
  img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # convert BGR(opencv format) to HSV format
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # cv.imread(img_dir, 0)
  img_gray_filter = cv.GaussianBlur(img_gray,(5,5),0)
  img_adaptive_mean = cv.adaptiveThreshold(img_gray_filter,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,21,2)
  img_adaptive_mean = cv.cvtColor(img_adaptive_mean, cv.COLOR_BGR2RGB)

  k = {'img' : img,
       'img_hsv' : img_hsv,
       'img_gray' : img_gray,
       'img_gray_filter' : img_gray_filter,
       'img_adaptive_mean' : img_adaptive_mean}

  return img_hsv, img_adaptive_mean, k