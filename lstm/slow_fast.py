


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, MaxPooling3D, Dropout, Reshape
from tensorflow.keras.layers import Conv3D, Dense, GlobalAveragePooling3D, Input, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import argparse
from sklearn.metrics import accuracy_score,confusion_matrix




parser = argparse.ArgumentParser(description="Train SlowFast")
parser.add_argument('-ow',type=str,help= 'name to store the weights in s3',required=True)
parser.add_argument('-w',type=str,help= 'name of the weights file in s3',required=False)
args = parser.parse_args()





def load_model_weights(model,weights_file_name):
    """Loads the model weights from s3 into the model"""
    import boto3
    import pickle
    s3 = boto3.resource('s3')
    my_pickle = pickle.loads(s3.Bucket("thesis-videos-dlb").Object(weights_file_name).get()['Body'].read())
    model.set_weights(my_pickle)
    return model





# In case of doubt with resnets check https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c
def identity_block(input_tensor, kernel_size, filters, stage, block, path, non_degenerate_temporal_conv=False):
    """So no convolutional layer in the shortcut. That means that the input has the
    same dimension as the output

    Arguments:
        input_tensor: input tensor
        kernel_size: The kernel size of the the middle conv layer at main path
        stage: integer, used for naming
        block: 'a','b'.... for naming

    Returns:
        Output tensor for the block
    """

    filters1, filters2, filters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1
    conv_name_base = str(path) + 'res' + str(stage) + block + '_branch'
    bn_name_base = str(path) + 'bn' + str(stage) + block + '_branch'
    # bn stands for batch normalization

    if non_degenerate_temporal_conv == True:
        x = Conv3D(filters1, (3, 1, 1), padding='same', kernel_regularizer=l2(1e-4), name=conv_name_base + '2a')(
            input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

    else:
        x = Conv3D(filters1, (1, 1, 1), kernel_regularizer=l2(1e-4), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size, padding='same', kernel_regularizer=l2(1e-4), name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1), padding='same', kernel_regularizer=l2(1e-4), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    """No convolution in the shortcut"""
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, path, strides=(1, 2, 2),
               non_degenerate_temporal_conv=False):
    """In this resnet there is a convolution on the shortcut. So we will have the same blocks as in the identity blocks
    plus the convolution in the shortcut. Note that the size of the input_tensor and the output don't match"""

    filters1, filters2, filters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1

    conv_name_base = str(path) + 'res' + str(stage) + block + '_branch'
    bn_name_base = str(path) + 'bn' + str(stage) + block + '_branch'

    if non_degenerate_temporal_conv == True:
        x = Conv3D(filters1, (3, 1, 1), strides=strides, padding='same', kernel_regularizer=l2(1e-4),
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

    else:
        x = Conv3D(filters1, (1, 1, 1), strides=strides, padding='same', kernel_regularizer=l2(1e-4),
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size, padding='same', kernel_regularizer=l2(1e-4), name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1), padding='same', kernel_regularizer=l2(1e-4), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv3D(filters3, (1, 1, 1), strides=strides, kernel_regularizer=l2(1e-4), name=conv_name_base + '1')(
        input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x



def lateral_connection(fast_res_block, slow_res_block, stage, method='T_conv', alpha=8, beta=1 / 8):
    lateral_name = 'lateral' + '_stage_' + str(stage)
    connection_name = 'connection' + '_stage_' + str(stage)
    if method not in ['T_conv', 'T_sample', 'TtoC_sum', 'TtoC_concat']:
        raise ValueError("method should be one of ['T_conv','T_sample','TtoC_sum','TtoC_concat']")

    if method == 'T_conv':
        lateral = Conv3D(int(2 * beta * int(fast_res_block.shape[4])), padding='same', kernel_size=(5, 1, 1),
                         strides=(int(alpha), 1, 1), kernel_regularizer=l2(1e-4), name=lateral_name)(fast_res_block)
        connection = Concatenate(axis=-1, name=connection_name)([slow_res_block, lateral])

    if method == 'T_sample':
        def sample(input, stride):
            return tf.gather(input, tf.range(0, input.shape[1], stride), axis=1)

        lateral = Lambda(sample, arguments={'stride': alpha}, name=lateral_name)(fast_res_block)
        connection = Concatenate(axis=-1, name=connection_name)([slow_res_block, lateral])

    if method == 'TtoC_concat':
        lateral = Reshape((int(int(fast_res_block.shape[1]) / alpha), int(fast_res_block.shape[2]),
                           int(fast_res_block.shape[3]), int(alpha * fast_res_block.shape[4])), name=lateral_name)(
            fast_res_block)
        connection = Concatenate(axis=-1, name=connection_name)([slow_res_block, lateral])

    if method == 'TtoC_sum':
        if alpha * beta != 1:
            raise ValueError('The product of alpha and beta must equal 1 in TtoC_sum method')
        lateral = Reshape((int(int(fast_res_block.shape[1]) / alpha), int(fast_res_block.shape[2]),
                           int(fast_res_block.shape[3]), int(alpha * fast_res_block.shape[4])), name=lateral_name)(
            fast_res_block)
        connection = Add(name=connection_name)([slow_res_block, lateral])

    return connection


def SlowFast(clip_shape=[20, 220, 220, 3], num_class=3, alpha=8, beta=1 / 8, tau=16, method='T_conv'):
    """Instantiates the SF

    Arguments:
        clip_shape: shapre of the video_clip
        num_class: numbers of video class
        alpha: Sample density
        beta: Channel capacity
        tau: nº frames sampled by the Slow pathway
        method = one of ['T_conv','T_sample','TtoC_sum','TtoC_concat'] mentioned in paper


    Returns:
        A keras model

    Raises:
        ValueError: in case of invalid argument for 'method' """

    clip_input = Input(shape=clip_shape)

    def data_layer(input, stride):
        return tf.gather(input, tf.range(0, 20, stride), axis=1)

    # Lambda layers act as a wrapper. In this case with data_layer and input
    slow_input = Lambda(data_layer, arguments={'stride': tau}, name='slow_input')(clip_input)
    fast_input = Lambda(data_layer, arguments={'stride': int(tau / alpha)}, name='fast_input')(clip_input)
    print('slow_path_input_shape', slow_input.shape)
    print('fast_path_input_shape', fast_input.shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1

    # -- fast pathway ---
    x_fast = Conv3D(8, (5, 7, 7), strides=(1, 2, 2), padding='same', kernel_regularizer=l2(1e-4), name='fast_conv1')(
        fast_input)
    x_fast = BatchNormalization(axis=bn_axis, name='fast_bn_conv1')(x_fast)
    x_fast = Activation('relu')(x_fast)
    pool1_fast = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), name='poo1_fast')(x_fast)

    x_fast = conv_block(pool1_fast, [1, 3, 3], [int(64 * beta), int(64 * beta), int(256 * beta)], stage=2, block='a',
                        path='fast', strides=(1, 1, 1), non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(64 * beta), int(64 * beta), int(256 * beta)], stage=2, path='fast',
                            block='b', non_degenerate_temporal_conv=True)
    res2_fast = identity_block(x_fast, [1, 3, 3], [int(64 * beta), int(64 * beta), int(256 * beta)], stage=2,
                               path='fast', block='c', non_degenerate_temporal_conv=True)

    x_fast = conv_block(res2_fast, [1, 3, 3], [int(128 * beta), int(128 * beta), int(512 * beta)], stage=3, path='fast',
                        block='a', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(128 * beta), int(128 * beta), int(512 * beta)], stage=3,
                            path='fast', block='b', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(128 * beta), int(128 * beta), int(512 * beta)], stage=3,
                            path='fast', block='c', non_degenerate_temporal_conv=True)
    res3_fast = identity_block(x_fast, [1, 3, 3], [int(128 * beta), int(128 * beta), int(512 * beta)], stage=3,
                               path='fast', block='d', non_degenerate_temporal_conv=True)

    x_fast = conv_block(res3_fast, [1, 3, 3], [int(256 * beta), int(256 * beta), int(1024 * beta)], stage=4,
                        path='fast', block='a', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(256 * beta), int(256 * beta), int(1024 * beta)], stage=4,
                            path='fast', block='b', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(256 * beta), int(256 * beta), int(1024 * beta)], stage=4,
                            path='fast', block='c', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(256 * beta), int(256 * beta), int(1024 * beta)], stage=4,
                            path='fast', block='d', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(256 * beta), int(256 * beta), int(1024 * beta)], stage=4,
                            path='fast', block='e', non_degenerate_temporal_conv=True)
    res4_fast = identity_block(x_fast, [1, 3, 3], [int(256 * beta), int(256 * beta), int(1024 * beta)], stage=4,
                               path='fast', block='f', non_degenerate_temporal_conv=True)

    x_fast = conv_block(res4_fast, [1, 3, 3], [int(512 * beta), int(512 * beta), int(2048 * beta)], stage=5,
                        path='fast', block='a', non_degenerate_temporal_conv=True)
    x_fast = identity_block(x_fast, [1, 3, 3], [int(512 * beta), int(512 * beta), int(2048 * beta)], stage=5,
                            path='fast', block='b', non_degenerate_temporal_conv=True)
    res5_fast = identity_block(x_fast, [1, 3, 3], [int(512 * beta), int(512 * beta), int(2048 * beta)], stage=5,
                               path='fast', block='c', non_degenerate_temporal_conv=True)

    # -- slow pathway ---

    x = Conv3D(64, (1, 7, 7), strides=(1, 2, 2), padding='same', kernel_regularizer=l2(1e-4), name='slow_conv1')(
        slow_input)
    x = BatchNormalization(axis=bn_axis, name='slow_bn_conv1')(x)
    x = Activation('relu')(x)
    pool1 = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), name='poo1_slow')(x)
    pool1_conection = lateral_connection(pool1_fast, pool1, stage=1, method=method, alpha=alpha, beta=beta)

    x = conv_block(pool1_conection, [1, 3, 3], [64, 64, 256], stage=2, block='a', strides=(1, 1, 1), path='slow')
    x = identity_block(x, [1, 3, 3], [64, 64, 256], stage=2, block='b', path='slow')
    res2 = identity_block(x, [1, 3, 3], [64, 64, 256], stage=2, block='c', path='slow')
    res2_conection = lateral_connection(res2_fast, res2, stage=2, method=method, alpha=alpha, beta=beta)

    x = conv_block(res2_conection, [1, 3, 3], [128, 128, 512], stage=3, block='a', path='slow')
    x = identity_block(x, [1, 3, 3], [128, 128, 512], stage=3, block='b', path='slow')
    x = identity_block(x, [1, 3, 3], [128, 128, 512], stage=3, block='c', path='slow')
    res3 = identity_block(x, [1, 3, 3], [128, 128, 512], stage=3, block='d', path='slow')
    res3_conection = lateral_connection(res3_fast, res3, stage=3, method=method, alpha=alpha, beta=beta)

    x = conv_block(res3_conection, [1, 3, 3], [256, 256, 1024], stage=4, block='a', path='slow',
                   non_degenerate_temporal_conv=True)
    x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='b', path='slow',
                       non_degenerate_temporal_conv=True)
    x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='c', path='slow',
                       non_degenerate_temporal_conv=True)
    x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='d', path='slow',
                       non_degenerate_temporal_conv=True)
    x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='e', path='slow',
                       non_degenerate_temporal_conv=True)
    res4 = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='f', path='slow',
                          non_degenerate_temporal_conv=True)
    res4_conection = lateral_connection(res4_fast, res4, stage=4, method=method, alpha=alpha, beta=beta)

    x = conv_block(res4_conection, [1, 3, 3], [512, 512, 2048], stage=5, block='a', path='slow',
                   non_degenerate_temporal_conv=True)
    x = identity_block(x, [1, 3, 3], [512, 512, 2048], stage=5, block='b', path='slow',
                       non_degenerate_temporal_conv=True)
    res5 = identity_block(x, [1, 3, 3], [512, 512, 2048], stage=5, block='c', path='slow',
                          non_degenerate_temporal_conv=True)

    fast_output = GlobalAveragePooling3D(name='avg_pool_fast')(res5_fast)
    slow_output = GlobalAveragePooling3D(name='avg_pool_slow')(res5)
    concat_output = Concatenate(axis=-1)([slow_output, fast_output])
    concat_output = Dropout(0.5)(concat_output)
    output = Dense(num_class, activation='softmax', kernel_regularizer=l2(1e-4), name='fc')(concat_output)

    # Create model.
    inputs = clip_input
    output = output
    model = Model(inputs, output, name='slowfast_resnet50')

    return model



def load_processed_data():
    import numpy as np
    X_train = np.load('./processed_data/X_train_NO_features_AUG.npy',allow_pickle=True)
    y_train = np.load('./processed_data/Y_train_NO_features_AUG.npy', allow_pickle=True)
    X_test = np.load('./processed_data/X_test_NO_features_AUG.npy', allow_pickle=True)
    y_test = np.load('./processed_data/Y_test_NO_features_AUG.npy',allow_pickle=True)
    """Temporal FIX HERE, IF THERE ARE X MISSLABELED OBSERVATIONS IS OK """
    X_train = [X_train[i][:20] for i in range(len(X_train)) if X_train[i].shape >= (20, 224, 224, 3)]
    X_train = np.stack(X_train)
    y_train = y_train[:len(X_train)]
    X_test = [X_test[i][:20] for i in range(len(X_test)) if X_test[i].shape >= (20, 224, 224, 3)]
    X_test = np.stack(X_test)
    y_test = y_test[:len(X_test)]
    return X_train,y_train,X_test,y_test



def save_results(score,confusion):
    import json
    with open('checkpoints/SlowFastResults', 'w') as file:
        json.dump(history.history, file)
        file.write(str(score))
        file.write(str(confusion))


if __name__ == '__main__':
    X_train , y_train, X_test, y_test = load_processed_data()
    model = SlowFast(clip_shape=[20, 224, 224, 3], num_class=3, alpha=8, beta=1 / 8, tau=16, method='T_conv')
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    if args.w == True:
        model = load_model_weights(model,args.w)
    history = model.fit(X_train,y_train,batch_size=24,epochs=3)
    prediction = model.predict(X_test)
    predicted_classes = [np.argmax(i) for i in prediction]
    real_classes = [np.argmax(i) for i in y_test]
    score = accuracy_score(real_classes, predicted_classes)
    confusion = confusion_matrix(real_classes, predicted_classes)
    save_results(score,confusion)


