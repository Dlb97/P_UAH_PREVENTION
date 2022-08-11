import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.optimizers import Adam
import SSL_functions as f
import argparse


parser = argparse.ArgumentParser(description="Train Two Stream Network Model")
parser.add_argument('-video',type=str,help='path to video',required=True)
parser.add_argument('-weights',type=str,help='path to load and save weights',required=True)
args = parser.parse_args()



output_shape = 1
dropout_ratio = 0.9


def temporal_stream(output_shape,dropout_ratio):
    """
    Creates the temporal Stream
    """
    input_temporal = layers.Input(name="input_temporal",shape=(224,224,6))
    x = layers.Conv2D(filters=96, kernel_size=(7,7),strides=(2,2))(input_temporal)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2),name="max")(x)
    x = layers.Conv2D(filters=256, kernel_size=(5,5),strides=(2,2))(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3),strides=(1,1))(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3),strides=(1,1))(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3),strides=(1,1))(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Dense(4096)(x)
    x = layers.Dropout(dropout_ratio)(x)
    x = layers.Dense(2048)(x)
    x = layers.Dropout(dropout_ratio)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(output_shape,activation='sigmoid')(x)
    temporal_stream = Model(input_temporal,output,name="temporal_stream")

    return temporal_stream

#python3 SSTwo_Stream.py -video /Users/david/workspace/thesis/PREVENTION-DATASET/video_camera1.mp4 -weights ./checkpoints/TS.ckpt

if __name__ == '__main__':
    temporal_stream = temporal_stream(output_shape,dropout_ratio)
    temporal_stream.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )


    if args.weights:
        print('Loading Weights')
        temporal_stream.load_weights(args.weights)


    checkpoints_TS_model = f.define_checkpoint(args.weights)
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_test_frame = round((total_frames * 0.7))


    #Process both train and evaluate test to be able to perform the checkpoints
    obs,labels = f.process_video_two_stream(cap, 0, start_test_frame)
    test_obs, test_labels = f.process_video_two_stream(cap, start_test_frame, total_frames - 100)

    temporal_stream.fit(obs, labels, validation_data=(test_obs,test_labels) ,epochs=3, callbacks=[checkpoints_TS_model])
