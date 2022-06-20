
import argparse
parser = argparse.ArgumentParser(description="Save to s3 only the LC from a specific video of the prevention dataset")
parser.add_argument('-cp' ,type=str ,help='path to the video' ,required=True)
parser.add_argument('-dp' ,type=str ,help='path to the detections file' ,required=True)
parser.add_argument('-lp' ,type=str ,help='path to the lane changes file' ,required=True)
parser.add_argument('-tte' ,type=int ,help='obs_horizon' ,required=True)
parser.add_argument('-o' ,type=str ,help='name of file to store results', required=True)
args = parser.parse_args()


import cv2
import numpy as np
import boto3
import pandas as pd


def read_lane_changes(file_path ,caption_path ,obs_horizon ,tte):
    import cv2
    """Reads the lane_change file that stores the frames that contain a lc
    Arguments:
        file_path: path to the lane change file
        caption: path to the video file
    Returns:
         all_frames:
         lane_changes
    """
    cap = cv2.VideoCapture(caption_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frame s ={k :0 for k in range(frame_count)}
    lane_changes = {}
    id_counter = 1
    with open(file_path) as file:
        for line in file.readlines():
            line = line.strip('\n').split(' ')
            lc_class = int(float(line[2]))
            object = int(float(line[1]))
            start = int(float(line[3]))
            middle = int(float(line[4]))
            end = int(float(line[5]))
            blinker = int(float(line[6]))
            if start != -1:
                all_frames = fill_range_frames(all_frames ,obs_horizon ,tte ,start ,id_counter)
                lane_changes[id_counter] = create_a_lane(lc_class ,start ,middle ,end ,blinker ,object)
                id_counter += 1

    return all_frames ,lane_changes




def create_a_lane(lc_class ,start ,middle ,end ,blinker ,object):
    """Creates a lane that will be added to the lane_changes dict. It is used in the read_lane_changes function.
    The conversion to float and int solves the issue with the starting 0
    """
    import numpy as np
    template_lane = {'start': None, 'middle': None, 'end': None, 'blinker': 0 ,'lc_class' :0 ,'object_id' :None}
    template_lane['start'] = start
    template_lane['middle'] = middle
    template_lane['end'] =  end
    template_lane['blinker'] = blinker
    template_lane['object_id'] = object
    if lc_class == 3:
        template_lane['lc_class'] = np.array([0 ,1 ,0])
    else:
        template_lane['lc_class'] = np.array([0 ,0 ,1])

    return template_lane





def fill_range_frames(all_frames ,obs_horizon ,tte ,start ,id_counter):
    """Fills the entire range in the all_frames dictionary of a clip that contains a lane change.
    So for new dataset call with 0 obs and 20 tte"""

    point = start - obs_horizon
    limit = start + tte

    while point <= limit:
        all_frames[point] = id_counter
        point += 1
    return all_frames



def create_tracker_dictionary(file_path):
    """
    FOR DOTS
    Reads the file that contains the object detection coordinates and returns a dict.
    Returns:
        tracker_dict :  object_id : {'frame_id': (x,y,w,h) ,'frame_id': (x,y,w,h)  }
    """
    tracker_dict = {}
    with open(file_path) as file:
        for line in file.readlines():
            line = line.strip('\n').split(' ')
            object_id = int(line[1])
            if object_id < 0:
                object_id = object_id * -1
            frame = int(line[0])
            roi = create_roi(line)
            if object_id not in tracker_dict.keys():
                tracker_dict[object_id] = {}
            tracker_dict[object_id][frame] = roi
    return tracker_dict





def create_roi(line):
    """Used in create_tracker_dictionary. It is a template that will contain one detection so x and y are all the coordinates of the box"""
    template_box = {'coordinates': {'x': [], 'y': []}}
    coordinates = line[8:]
    template_box = get_all_coordinates(coordinates, template_box)
    min_x = min(template_box['coordinates']['x'])
    max_x = max(template_box['coordinates']['x'])
    min_y = min(template_box['coordinates']['y'])
    max_y = max(template_box['coordinates']['y'])
    w = max_x - min_x
    h = max_y - min_y
    frame_id = int(line[0])
    # roi = {frame_id: (min_x, min_y, w, h)}
    return (min_y, min_x, h, w)



def get_all_coordinates(coordinates, template_box):
    for i in range(len(coordinates)):
        if i % 2 == 0:
            template_box['coordinates']['x'].append(int(coordinates[i]))

        else:
            template_box['coordinates']['y'].append(int(coordinates[i]))

    return template_box



def grab_roi_from_cap(cap_path ,coordinates ,frame ,desired_size ,scale):
    """Grabs the ROI based on the frame, coordinates and context desired by directly looking at the video cap"""
    import cv2
    cap = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, img = cap.read()
    img = expand_roi(scale ,img ,coordinates)
    img = pad_image(desired_size ,img)
    return img


def expand_roi(scale ,img ,coordinates):
    """Expands the size of the ROI we will grab to take context information"""
    w = round((coordinates[2] * scale) /2)
    h = round((coordinates[3] * scale) / 2)
    w1 = coordinates[0] - w
    w2 = coordinates[0] + coordinates[2] + w
    h1 = coordinates[1] - h
    h2 = coordinates[1] + coordinates[3] + h
    # Check begining and end of x axis
    if w1 < 0:
        w1 = 0
    if w2 > img.shape[1]:
        w2 = img.shape[1]

    # Check begining and end of y axis
    if h1 < 0:
        h1 = 0
    if h2 > img.shape[0]:
        h2 = img.shape[0]

    return img[h1:h2, w1:w2]


def pad_image(desired_shape, img):
    """Pads an image according to the desired shape we want for the video"""
    import numpy as np
    w = img.shape[1]
    h = img.shape[0]
    if w > desired_shape[1]:
        img = img[:, :desired_shape[1]]
        a1 = 0
        a2 = 0
    else:
        a1 = round((desired_shape[1] - w) / 2)
        a2 = desired_shape[1] - w - a1

    if h > desired_shape[0]:
        img = img[:desired_shape[0], :]
        b1 = 0
        b2 = 0
    else:
        b1 = round((desired_shape[0] - h) / 2)
        b2 = desired_shape[0] - h - b1

    padded_image = np.pad(img, ((b1, b2), (a1, a2), (0, 0)), 'constant')

    return padded_image


def save_individual_video(s3_folder, video_name, frame, local_path, video, desired_size):
    """Saves the video locally and to s3
    INPUTS:
        - s3_folder: name of the s3 folder where videos should be stored
        - object: the object id that will be included in the name
        - local_path: Local path to folder where the videos would be stored  CRATE IT BEFORE RUNNING
        - videos: array of shape (224,224,3)
        - Length of the videos

    OUTPUTS:
        - video_names : LIST with the video names that will be later used to build the CSV file that contains path and label
    """
    import cv2
    import boto3
    s3 = boto3.client('s3')
    fps = 20
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
    vout = cv2.VideoWriter()
    key_name = video_name + '_' + str(frame) + '.mp4'
    success = vout.open(local_path + key_name, fourcc, fps, desired_size, True)
    for frame in range(len(video)):
        vout.write(video[frame])
    vout.release()
    cv2.destroyAllWindows()
    s3_key = s3_folder + '/' + key_name
    s3.upload_file(local_path + key_name, Bucket='thesis-videos-dlb', Key=s3_key)
    return s3_key


def save_info(file_name, video_name, label):
    """Writes the s3 path and the label in a txt file for later training purposes"""
    try:
        with open(file_name, 'a') as file:
            file.write(video_name)
            file.write(',')
            file.write(label)
            file.write('\n')
    except FileNotFoundError:
        with open(file_name, 'w') as file:
            file.write(video_name)
            file.write(',')
            file.write(label)
            file.write('\n')


def save_videos_only_lc(lane_changes, detections, cap_path, output_file):
    for k, v in lane_changes.items():
        object_id = v['object_id']
        label = str(v['lc_class'])
        start = v['start']
        object_filtered = {k: v for k, v in detections[object_id].items() if (k >= start and k <= start + 20)}
        try:
            images = np.stack([grab_roi_from_cap(cap_path, v, k, (224, 224), 2) for k, v in object_filtered.items()],
                              axis=0)
            v_name = save_individual_video('lc-only', str(object_id), start,
                                           '/Users/david/workspace/thesis/PREVENTION-DATASET/positive_samples/', images,
                                           (224, 224))
            save_info(output_file, v_name, label)
        except ValueError:
            print('Error on lane change: ', k)


if __name__ == '__main__':
    all_frames, lane_changes = read_lane_changes(args.lp, args.cp, 0, args.tte)
    detections = create_tracker_dictionary(args.dp)
    save_videos_only_lc(lane_changes, detections, args.dp, args.o)

