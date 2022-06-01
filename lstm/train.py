
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import os
from lstm.functions import *
from sklearn.metrics import accuracy_score, confusion_matrix
from lstm.models import *



"""Independently of the model used this cell has to run to prepare the class balanced dataset"""

lc, no_lc = get_balanced_dataset()
df = pd.concat([lc,no_lc])
df = df.reset_index()
X_train, X_test, y_train, y_test = train_test_split(df["path"],df["label"],test_size=0.3,random_state=19)

train = pd.concat([X_train,y_train],axis=1)
test = pd.concat([X_test,y_test],axis=1)

train = filter_videos(train)
test = filter_videos(test)





"""Decide the model to use  for training here choose one out  of the 2 next cells"""
##


X_train,y_train = prepare_input_without_feature_extractor(train)
X_test,y_test = prepare_input_without_feature_extractor(test)
model = action_model_LSTM()
optimizer = keras.optimizers.Adam(0.001)
model.compile(optimizer,'categorical_crossentropy',metrics=['acc'])
model.fit(X_train,y_train,epochs=10)
##

feature_extractor = build_feature_extractor()
X_train, y_train = prepare_all_videos(train,feature_extractor)
model = get_sequence_model(train)
model.fit([X_train[0],X_train[1]],y_train,epochs=10)


##
X_test, y_test = prepare_all_videos(test,feature_extractor)
prediction = model.predict([X_test[0],X_test[1]])
predicted_classes = [np.argmax(i) for i in prediction ]
score = accuracy_score(y_test,predicted_classes)
print(confusion_matrix(y_test,predicted_classes))