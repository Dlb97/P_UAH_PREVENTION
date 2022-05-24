
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
