import SSL_functions as f
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


#python3 -video /Users/david/workspace/thesis/PREVENTION-DATASET/video_camera1.mp4 -weights /Users/david/workspace/thesis/thesis_repo/P_UAH_PREVENTION/SSL/checkpoints/cp.ckpt


target_shape = (224, 224)



import argparse
parser = argparse.ArgumentParser(description="Train Siamese Model")
parser.add_argument('-video',type=str,help='path to video',required=True)
parser.add_argument('-weights',type=str,help='path to load and save weights',required=True)
args = parser.parse_args()



if __name__ == '__main__':

    #CREATE NETWORK
    embedding = f.create_base_cnn()
    order_network = f.create_Binary_order_prediction_network(embedding)
    SiameseModel = f.SiameseModel(order_network)
    SiameseModel.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    #LOAD WEIGHTS & DEFINE CHECKPOINT

    if args.weights:
        print('Loading Weights')
        SiameseModel.load_weights(args.weights)

    checkpoints_Binary_model = f.define_checkpoint(args.weights)

    caption_path = args.video
    cap = cv2.VideoCapture(caption_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_test_frame = round((total_frames * 0.8))

    #Process both train and evaluate test to be able to perform the checkpoints
    a, p, n, labels = f.process_video(cap, 0, start_test_frame)
    test_a, test_p, test_n, test_labels = f.process_video(cap, start_test_frame, total_frames - 100)

    #Save the best model based on the val accuracy
    SiameseModel.fit([a, p, n], labels, validation_data=([test_a,test_p,test_n],test_labels) ,epochs=3, callbacks=[checkpoints_Binary_model])




