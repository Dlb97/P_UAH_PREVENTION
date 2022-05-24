
import cv2
import numpy as np
from tensorflow import keras


def get_csv(file):
    import boto3
    import pandas as pd
    import io
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='thesis-videos-dlb',Key='csv/' + file)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df




def get_video(o):
    import boto3
    import cv2
    s3 = boto3.client('s3')
    print(o)
    try:
        obj = s3.get_object(Bucket='thesis-videos-dlb', Key=o)
        bytes_content = obj['Body'].read()
        filename = 'temporary_video.avi'
        with open(filename,'wb') as file:
            file.write(bytes_content)
        file.close()
        cap = cv2.VideoCapture(filename)
        return cap
    except Exception:
        return




"""------------------------------------------SECTION FROM ARTICLE ---------------------------------------------------------"""

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = get_video(path) #EDITED HERE TO GRAB MY VIDEO FROM S3
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        if cap:
            cap.release()
    return np.array(frames)



def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((224, 224, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def prepare_single_video(frames,num_features=2048):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, 20,), dtype="bool")
    frame_features = np.zeros(shape=(1, 20, num_features), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(20, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def prepare_all_videos(df,num_features=2048):
    num_samples = len(df)
    video_paths = df["path"].values.tolist()

    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["label"]))
    labels = df["label"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, 20), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, 20, num_features), dtype="float32")

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(path)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, 20,), dtype="bool")
        temp_frame_features = np.zeros(shape=(1, 20, num_features), dtype="float32")

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(20, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


def get_sequence_model(df,num_features=2048):

    #Added here
    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["label"]))
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

    rnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return rnn_model


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