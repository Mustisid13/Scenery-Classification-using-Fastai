import streamlit as st
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.basic_train import load_learner
import PIL.Image

import io
import torch

st.title("Scenery Classification")

st.set_option('deprecation.showfileUploaderEncoding', False)

file_buffer = st.file_uploader("", type=["jpg","png","jpeg"])
if file_buffer!=None:
	temp = PIL.Image.open(file_buffer).resize((224,224))
	image = open_image(file_buffer)

	st.image(temp,caption="uploaded Image")

	st.write("Prediction")

	pred_class,pred_idx,outputs = learner.predict(image)
	st.write(f" {pred_class}")
