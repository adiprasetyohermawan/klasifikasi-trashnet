import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

from pathlib import Path
from preprocessing_img import preprocessing_data
from feature_extraction import hu_moment, color_moment
from inference import predict

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pickle

st.set_page_config(page_title="Trashnet Prediction Dashboard", layout="wide")

st.write("""
# Trash Image Prediction Dashboard
Created by : [@Hermawan Adi Prasetyo](https://www.linkedin.com/in/hermawan-adi-prasetyo/)
""")

def plot_img(**k):
  st.image(k["img"], caption='RGB Image', width=500)
  st.image(k["img_hsv"], caption='HSV Image', width=500)
  st.image(k["img_gray"], caption='Grayscale Image', width=500)
  st.image(k["img_gray_filter"], caption='Filtered Grayscale Image', width=500)
  st.image(k["img_adaptive_mean"], caption='Binary Image', width=500)

def table_dataframe(df):
   st.dataframe(df, use_container_width=True)

def trash():
    st.write("""
    This app uses a machine learning model to predict images of **anorganic** and **recyclable trash**, classifying them into five categories: **Cardboard**, **Glass**, **Metal**, **Paper**, and **Plastic**.
    
    The data is sourced from the [Trashnet Dataset](https://github.com/garythung/trashnet) by Gary Thung & Mindy Yang.
    """)
    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your trash image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.sidebar.header("Upload Succeed")
        # img_dir = uploaded_file._file_urls.upload_url
        # img_path = Path(str(img_dir)).resolve()

        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        img_hsv, img_adaptive_mean, k = preprocessing_data(bytes_data)
        plot_img(**k)

        df_7huMoment = hu_moment(img_adaptive_mean)
        st.header("7 Hu's Moment Invariant Shape Feature Values", divider="gray")
        table_dataframe(df_7huMoment)

        df_9hsvMoment = color_moment(img_hsv)
        st.header("9 HSV Color Moment Feature Values", divider="gray")
        table_dataframe(df_9hsvMoment)
        
        prediction_result = predict(df_9hsvMoment)
        st.write(f"The trash image is predicted as **{prediction_result}** class !!!")

trash()