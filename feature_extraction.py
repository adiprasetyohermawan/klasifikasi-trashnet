import cv2 as cv
import numpy as np
import pandas as pd

"""
Ekstraksi Fitur Bentuk dengan metode Hu's Moment Invariants
"""
def hu_moment(img_binary):
  dump = []
  gray_img = cv.cvtColor(img_binary,cv.COLOR_BGR2GRAY) # Mengubah ke 1 Kanal
  hu = cv.HuMoments(cv.moments(gray_img))
  for i in range(0, 7):
    hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))
  hu = hu.reshape((1, 7)).tolist()[0]
  dump.append(hu)
  cols = ["Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"]
  dataframe = pd.DataFrame(dump, columns=cols)
  return dataframe


"""
Ekstraksi Fitur Warna dengan metode HSV Color Moments
"""
def color_moment(img_hsv):
  # Split the channels - h,s,v
  h, s, v = cv.split(img_hsv)
  # Initialize the color feature
  color_feature = []
  # N = h.shape[0] * h.shape[1]
  # The first central moment - average
  h_mean = np.mean(h)  # np.sum(h)/float(N)
  s_mean = np.mean(s)  # np.sum(s)/float(N)
  v_mean = np.mean(v)  # np.sum(v)/float(N)
  color_feature.extend([h_mean, s_mean, v_mean])
  # The second central moment - standard deviation
  h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
  s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
  v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
  color_feature.extend([h_std, s_std, v_std])
  # The third central moment - the third root of the skewness
  h_skewness = np.mean(abs(h - h.mean())**3)
  s_skewness = np.mean(abs(s - s.mean())**3)
  v_skewness = np.mean(abs(v - v.mean())**3)
  h_thirdMoment = h_skewness**(1./3)
  s_thirdMoment = s_skewness**(1./3)
  v_thirdMoment = v_skewness**(1./3)
  color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
  cols = ["meanH", "meanS", "meanV", "stdH", "stdS", "stdV", "skewH", "skewS", "skewV"]
  dataframe = pd.DataFrame([color_feature], columns=cols)
  return dataframe