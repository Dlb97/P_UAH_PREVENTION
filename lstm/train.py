
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import os

##

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


##
df = get_csv('v1')

##
df = get_csv('v10')
##

video_name = 'v1/2_0.mp4'
video = load_video(video_name)
feature_extractor = build_feature_extractor()
features,mask = prepare_single_video(video)
subset = df[0:300]
X_train, y_train = prepare_all_videos(subset)
model = get_sequence_model(subset)

##

model.fit([X_train[0],X_train[1]],y_train,epochs=4)