



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

import argparse
parser = argparse.ArgumentParser(description="Train a specific ML model on the UAH PREVENTION DATASET and store its results")
parser.add_argument('-m',type=str,help='Model to use',required=True)
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

    lc, no_lc = get_balanced_dataset()
    df = pd.concat([lc, no_lc])
    df = df.reset_index()
    X_train, X_test, y_train, y_test = train_test_split(df["path"], df["label"], test_size=0.3, random_state=19)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train = filter_videos(train)
    test = filter_videos(test)



    X_train, y_train = prepare_input_without_feature_extractor(train)
    X_test, y_test = prepare_input_without_feature_extractor(test)


    if args.aug:

        X_train, y_train = augment_dataset(X_train,y_train)




    """Perform feature extraction"""

    if args.f:

        X_train = extract_features(X_train)
        X_test = extract_features(X_test)
        y_train = prepare_labels(y_train)
        y_test = prepare_labels(y_test)





    """Train model and save results to S3"""

    model = get_model(args.m,train)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    checkpoint_filepath = 'checkpoints/checkpoint_' + args.m
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                   monitor='val_accuracy', mode='max',
                                                                   save_best_only=True)

    """Need to pass both features and mask in the case we extracted them"""
    if args.f :
        history = model.fit([X_train[0],X_train[1]],y_train, callbacks=[model_checkpoint_callback], validation_data=[X_test, y_test],
                        epochs=args.e, batch_size=args.b)
    else:
        history = model.fit(X_train, y_train, callbacks=[model_checkpoint_callback], validation_data=[X_test, y_test],epochs=args.e, batch_size=args.b)


    results_filename = 'results_' + args.m + '.txt'

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
    s3.upload_file(checkpoint_filepath + '.data-00000-of-00001' , 'thesis-videos-dlb', 'weights/' + '_weights_' + args.m)
    s3.upload_file('checkpoints/' + results_filename, 'thesis-videos-dlb', 'weights/' + '_results_' + args.m + '.txt')






