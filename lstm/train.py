



"""Code Logic:


TO DO LIST:

    - Modify prepare_one_video so that it is also possible to perform data augmentation when using the feature
      extractor ---- > Use the prepare one video with for loop that iterates through the array after the augmentation


Parser:
    - model to train (String it would also be the name to give to the results files)
    - data augmentation boolean . Performs data augmentation
    - feature extractor boolean  . Performs feature extraction
    - Epochs & batchsize


"""

#python3 train.py -m lstm -aug True -f False -e 6 -b 24 -c True
#python3 train.py -m lstm_extractor -aug True -f True -e 6 -b 32 -c True
#python3 train.py -m lstm_extractor -new True -fs True -n new_no_aug -aug False -f True -e 20 -b 32 -c True
#python3 train.py -m lstm -new True -fs False -n new_no_aug -aug False -f False -e 10 -b 32 -c True

import argparse
parser = argparse.ArgumentParser(description="Train a specific ML model on the UAH PREVENTION DATASET and store its results")
parser.add_argument('-m',type=str,help='Model to use',required=True)
parser.add_argument('-new',type=bool,help='Use New approach (Behavior recognition and not prediction)',required=True)
parser.add_argument('-fs',type=bool,help='Save the extracted features into np file?',required=True)
parser.add_argument('-n',type=str,help='Name for np files in case want ot save them',required=True)
parser.add_argument('-aug',type=bool,help='Perform data augmentation',required=True)
parser.add_argument('-f',type=bool,help='Use feature extractor',required=True)
parser.add_argument('-e',type=int,help='Number of epochs',required=True)
parser.add_argument('-b',type=int,help='Batch size',required=True)
parser.add_argument('-c',type=bool,help='Confussion matrix',required=True)
args = parser.parse_args()


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow import keras
    import pandas as pd
    import numpy as np
    import cv2
    import os
    #from lstm.functions import *
    from functions import *
    from sklearn.metrics import accuracy_score, confusion_matrix
    #from lstm.models import *
    from models import *
    import boto3
    import json

    """This section focuses on reading the csv that contains the path and label to each of the individual videos. Balancing
    the dataset so that there are a proportionate number of classes and reading the only presents in s3"""


    lc, no_lc = get_balanced_dataset(750)
    #Add PARSER HERE TO INCLUDE EXTRA VIDEOS + ONLY_LC. THOSE WILL BE OUR LC
    if args.new:
        extra_videos = pd.read_csv('../extra_videos.csv')
        only_lc = pd.read_csv('../ONLY_LC.txt',names=['path','label'],header=None)
        lc = pd.concat([only_lc, extra_videos])
    df = pd.concat([lc, no_lc])
    df = df.reset_index()
    X_train, X_test, y_train, y_test = train_test_split(df["path"], df["label"], test_size=0.2, random_state=19)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train = filter_videos(train)
    test = filter_videos(test)


    print('Loading Videos')
    X_train, y_train = prepare_input_without_feature_extractor(train)
    X_test, y_test = prepare_input_without_feature_extractor(test)
    if args.fs:
        np.save('./processed_data/X_train_NO_features_' + args.n + '.npy', X_train)
        np.save('./processed_data/X_test_NO_features_' + args.n + '.npy', X_test)
        np.save('./processed_data/Y_train_NO_features_' + args.n + '.npy', y_train)
        np.save('./processed_data/Y_test_NO_features_' + args.n + '.npy', y_test)


    if args.aug:

        X_train, y_train = augment_dataset(X_train,y_train)




    """Perform feature extraction"""

    if args.f:
        print('Performing feature extraction')
        X_train = extract_features(X_train)
        X_test = extract_features(X_test)
        y_train = prepare_labels(y_train)
        y_test = prepare_labels(y_test)

        if args.fs:
            print('Saving features')
            np.save('./processed_data/X_train_features_' + args.n + '.npy' , X_train[0])
            np.save('./processed_data/X_train_mask_' + args.n + '.npy', X_train[1])
            np.save('./processed_data/X_test_features_' + args.n + '.npy', X_test[0])
            np.save('./processed_data/X_test_mask_' + args.n + '.npy', X_test[1])
            np.save('./processed_data/y_train_features_' + args.n + '.npy', y_train)
            np.save('./processed_data/y_test_features_' + args.n + '.npy', y_test)



    """Train model and save results to S3"""

    model = get_model(args.m,train)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    checkpoint_filepath = 'checkpoints/checkpoint_' + args.m + args.n
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                   monitor='val_accuracy', mode='max',
                                                                   save_best_only=True)

    """Need to pass both features and mask in the case we extracted them"""
    if args.f :
        history = model.fit([X_train[0],X_train[1]],y_train, callbacks=[model_checkpoint_callback], validation_data=[X_test, y_test],
                        epochs=args.e, batch_size=args.b)
    else:
        history = model.fit(X_train, y_train, callbacks=[model_checkpoint_callback], validation_data=[X_test, y_test],epochs=args.e, batch_size=args.b)


    results_filename = 'results_' + args.m  + args.n + '.txt'

    with open('checkpoints/' + results_filename, 'w') as file:
        json.dump(history.history, file)
        if args.c:
            prediction = model.predict(X_test)
            predicted_classes = [np.argmax(i) for i in prediction]
            real_classes = [np.argmax(i) for i in y_test]
            score = accuracy_score(real_classes, predicted_classes)
            file.write(str(score))
            confusion = confusion_matrix(real_classes, predicted_classes)
            file.write(str(confusion))




    s3 = boto3.client('s3')
    s3.upload_file(checkpoint_filepath + '.data-00000-of-00001' , 'thesis-videos-dlb', 'weights/' + '_weights_' + args.m + args.n)
    s3.upload_file('checkpoints/' + results_filename, 'thesis-videos-dlb', 'weights/' + '_results_' + args.m + args.n + '.txt')






