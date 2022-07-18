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




"""MODEL:

BASE CNN ----> EMBEDDING --- > SIAMESE"""


def preprocess_image(image):
    """Process the image to convert it from numpy to tensor and return it in the appropiate size"""
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image,target_shape)
    return image




def preprocess_triplets(anchor,positive,negative):
    """Given the images prepocess them"""
    return (preprocess_image(anchor),preprocess_image(positive),preprocess_image(negative))




def get_APN_images(array):
    triplets = []
    for video in array:
        triplets.append(preprocess_triplets(video[0],video[7],video[19]))
    return triplets


def get_APN_images_as_arrays(array):
    anchors = []
    positive = []
    negative = []
    for video in array:
        anchors.append(video[0])
        positive.append(video[7])
        negative.append(video[19])
    return np.array(anchors),np.array(positive),np.array(negative)



def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])








"""HERE WE DON'T USE A ZIP DATASET AS WE NEED TO INPUT THE 'LABELS' AS WELL """






def create_base_cnn():
    base_cnn = resnet.ResNet50(weights="imagenet", input_shape=(224,224,3), include_top=False)

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(212,activation='relu')(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(76,activation='relu')(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(212)(dense2)


    embedding = Model(base_cnn.input,output,name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return embedding








def create_order_prediction_network(embedding):
    anchor_input = layers.Input(name="anchor",shape=target_shape + (3,))
    positive_input = layers.Input(name="positive",shape = target_shape + (3,))
    negative_input = layers.Input(name="negative",shape = target_shape + (3,))


    inputs_apn = layers.concatenate([
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input))]
    )

    x = layers.Dense(1,activation='sigmoid',name='output')(inputs_apn)
    order_network = Model(inputs=[anchor_input,positive_input,negative_input],outputs=x)

    order_network.summary()

    return order_network






class SiameseModel(Model):
    """The Order Prediction Model. Needs to be in a class to avoid the tf function error
    """

    def __init__(self,siamese_network):
        super(SiameseModel,self).__init__()
        self.siamese_network = siamese_network


    def call(self,inputs):
        return self.siamese_network(inputs)





def process_video(cap,start,end):
    a = []
    p = []
    n = []
    labels = []
    while start < end:
        try:
            v = get_video(cap,start)
            obs,obs_labels = prepare_observations(v)
            print(start + 20, 'Out of ', end)
            for i in obs:
                a.append(i[0])
                p.append(i[1])
                n.append(i[2])
            labels.extend(obs_labels)
            start += 20
        except IndexError:
            start += 20
            print('Custom Index error ')
            pass
    return np.array(a),np.array(p),np.array(n),np.array(labels)




def compute_optical_flow(prvs,next):
    """Returns the average optical flow from 2 consecutive frames"""
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow



def get_video(cap,start):
    """Returns a 20 frame video as a np array"""
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(20):
        success, img = cap.read()
        frames.append(cv2.resize(img,(224,224)))
    return np.stack(frames,axis=0)



def prepare_observations(v):
    optical_flows = [abs(np.mean(compute_optical_flow(v[i],v[i+1]))) for i in range(0,len(v)-1)]
    filtered_flows = remove_n_smallest(14,optical_flows)
    sampled_frames = filter_frames(v,optical_flows,filtered_flows)
    obs,labels = get_obs_and_label(sampled_frames)
    return obs,labels




def remove_n_smallest(n,items):
    """Removes the n smallest flows while preserving the object"""
    for _ in range(n):
        m = min(items)
        items = [i for i in items if i != m]
    return items



def filter_frames(v,all_flows,filtered_flows):
    """Returns those frames with the highest optical flow"""
    frames = []
    all_flows_dict = {str(i):idx for idx,i in enumerate(all_flows)}
    for i in filtered_flows:
        frames.append(v[all_flows_dict[str(i)]])
    return np.stack(frames,axis=0)



def get_obs_and_label(frames):
    """Prepare the observations and its labels according to the Shuffle and Learn paper"""
    positive_sample = (frames[1],frames[2],frames[3])
    negative_sample_one = (frames[1],frames[0],frames[3])
    negative_sample_two = (frames[1],frames[4],frames[3])
    obs = [positive_sample,negative_sample_one,negative_sample_two]
    labels = [1,0,0]
    return obs,labels


def define_checkpoint(path):
    checkpoint_filepath = path
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    return model_checkpoint_callback






if __name__ == '__main__':

    #CREATE NETWORK
    embedding = create_base_cnn()
    order_network = create_order_prediction_network(embedding)
    SiameseModel = SiameseModel(order_network)
    SiameseModel.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    #LOAD WEIGHTS & DEFINE CHECKPOINT

    if args.weights:
        print('Loading Weights')
        SiameseModel.load_weights(args.weights)

    checkpoints_Binary_model = define_checkpoint(args.weights)

    caption_path = args.video
    cap = cv2.VideoCapture(caption_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_test_frame = round((total_frames * 0.8))

    #Process both train and evaluate test to be able to perform the checkpoints
    a, p, n, labels = process_video(cap, 0, start_test_frame)
    test_a, test_p, test_n, test_labels = process_video(cap, start_test_frame, total_frames - 100)

    #Save the best model based on the val accuracy
    SiameseModel.fit([a, p, n], labels, validation_data=([test_a,test_p,test_n],test_labels) ,epochs=1, callbacks=[checkpoints_Binary_model])




