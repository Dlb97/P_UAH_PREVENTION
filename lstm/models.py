
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, TimeDistributed, Flatten, GRU, Dense, Dropout, LSTM

def build_convnet(shape=(224, 224, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3 ,3), input_shape=shape,
                     padding='same', activation='relu'))
    model.add(Conv2D(64, (3 ,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3 ,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3 ,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (3 ,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3 ,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(512, (3 ,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3 ,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    return model


def action_model_GRU(shape=(20,224, 224, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])

    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(GRU(64))
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model



def action_model_LSTM(shape=(20,224, 224, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])

    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(24, return_sequences=True,dropout=0.2))
    model.add(LSTM(24, return_sequences=True,dropout=0.2))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model



def testing_model():
    model = keras.Sequential()
    model.add(Conv2D(14, (3, 3), input_shape=(20,224, 224, 3),padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(10,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    return model



def GRU_extractor(num_features=2048):

    import numpy as np
    #label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["label"]))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((20, num_features))
    mask_input = keras.Input((20,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    #rnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return rnn_model



def get_model(model_name):
    models = {'lstm': action_model_LSTM() , 'gru': action_model_GRU(), 'gru_extractor': GRU_extractor(),
              'test': testing_model() }
    return models[model_name]