
import argparse
parser = argparse.ArgumentParser(description="Prepare the PREVENTION DATASET for a format compatible with the NN ")
parser.add_argument('-cp',type=str,help='path to the video',required=True)
parser.add_argument('-dp',type=str,help='path to the detections file',required=True)
parser.add_argument('-lp',type=str,help='path to the lane changes file',required=True)
parser.add_argument('-oh',type=int,help='obs_horizon',required=True)
parser.add_argument('-ds',nargs='+',type=int,help='desired size of the images',required=True)
parser.add_argument('-min',type=int,help='min n frames to create a video',required=True)
parser.add_argument('-s',type=int,help='context to take into account from ROI',required=True)
parser.add_argument('-rn',type=int,help='record number',required=False)
parser.add_argument('-op',type=str,help='path to store the resutls',required=False)
parser.add_argument('-s3',type=str,help='s3 folder',required=True)
parser.add_argument('-ns',type=bool,help='save big numpy video and label to s3',required=False)
parser.add_argument('-local',type=str,help='path to save the videos locally',required=True)

args = parser.parse_args()




"""-----------------------------------------LANE CHANGES SECTION ------------------------------------------------"""
def read_lane_changes(file_path,caption_path,obs_horizon,tte):
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
    all_frames={k:0 for k in range(frame_count)}
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
                all_frames = fill_range_frames(all_frames,obs_horizon,tte,start,id_counter)
                lane_changes[id_counter] = create_a_lane(lc_class,start,middle,end,blinker,object)
                id_counter += 1

    return all_frames,lane_changes



def create_a_lane(lc_class,start,middle,end,blinker,object):
    """Creates a lane that will be added to the lane_changes dict. It is used in the read_lane_changes function.
    The conversion to float and int solves the issue with the starting 0
    """
    import numpy as np
    template_lane = {'start': None, 'middle': None, 'end': None, 'blinker': 0,'lc_class':0,'object_id':None}
    template_lane['start'] = start
    template_lane['middle'] = middle
    template_lane['end'] =  end
    template_lane['blinker'] = blinker
    template_lane['object_id'] = object
    if lc_class == 3:
        template_lane['lc_class'] = np.array([0,1,0])
    else:
        template_lane['lc_class'] = np.array([0,0,1])

    return template_lane


def fill_range_frames(all_frames,obs_horizon,tte,start,id_counter):
    """Fills the entire range in the all_frames dictionary of a clip that contains a lane change."""

    point = start - obs_horizon
    limit = start + tte

    while point <= limit:
        all_frames[point] = id_counter
        point += 1
    return all_frames


"""-----------------------------------------END OF LANE CHANGES SECTION ------------------------------------------------"""


"""-----------------------------------------DETECTION SECTION ------------------------------------------------"""




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
    #roi = {frame_id: (min_x, min_y, w, h)}
    return (min_y, min_x, h, w)



def get_all_coordinates(coordinates, template_box):
    for i in range(len(coordinates)):
        if i % 2 == 0:
            template_box['coordinates']['x'].append(int(coordinates[i]))

        else:
            template_box['coordinates']['y'].append(int(coordinates[i]))

    return template_box





def grab_roi_from_cap(cap_path,coordinates,frame,desired_size,scale):
    """Grabs the ROI based on the frame, coordinates and context desired by directly looking at the video cap"""
    import cv2
    cap = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, img = cap.read()
    img = expand_roi(scale,img,coordinates)
    img = pad_image(desired_size,img)
    return img


def expand_roi(scale,img,coordinates):
    """Expands the size of the ROI we will grab to take context information"""
    w = round((coordinates[2] * scale) /2)
    h = round((coordinates[3] * scale) /2)
    w1 = coordinates[0] - w
    w2 = coordinates[0] + coordinates[2] + w
    h1 = coordinates[1] - h
    h2 = coordinates[1] + coordinates[3] + h
    #Check begining and end of x axis
    if w1 < 0:
        w1 = 0
    if w2 > img.shape[1]:
        w2 = img.shape[1]

    # Check begining and end of y axis
    if h1 < 0:
        h1 = 0
    if h2 > img.shape[0]:
        h2 = img.shape[0]


    return img[h1:h2,w1:w2]



def pad_image(desired_shape,img):
    """Pads an image according to the desired shape we want for the video"""
    import numpy as np
    w = img.shape[1]
    h = img.shape[0]
    if w > desired_shape[1]:
        img = img[:,:desired_shape[1]]
        a1 = 0
        a2 = 0
    else:
        a1 = round((desired_shape[1] - w) / 2)
        a2 = desired_shape[1]-w-a1

    if h > desired_shape[0]:
        img = img[:desired_shape[0], :]
        b1 = 0
        b2 = 0
    else:
        b1 = round((desired_shape[0] - h) / 2)
        b2 = desired_shape[0]-h-b1


    padded_image = np.pad(img, ((b1,b2), (a1,a2), (0, 0)), 'constant')

    return padded_image




def split_video(rois,obs_horizon):
    """Splits all the rois of a particular object into n equal size videos"""
    import numpy as np
    videos = []
    amount = round(len(rois) / obs_horizon)
    start = 0
    end = obs_horizon
    for i in range(amount):
        #There are twenty
        if len(rois[start:end]) == 20:
            subset = rois[start:end]
            videos.append(np.stack((subset),axis=0))
            start += 20
            end += 20

        #There are less than 20 hence we repeat frames
        else:
            subset = rois[start:]
            #print('Current length of the subset' ,len(subset))
            #print((amount * obs_horizon) - len(rois), 'Still to ad')
            for x in range( (amount * obs_horizon) - len(rois)):
                #Repeat the last roi
                subset.append(rois[-1])
            #print((len(subset)))
            videos.append(np.stack((subset), axis=0))
    return videos





def create_X_and_Y(cap_path,all_frames,lane_changes,tracker_dict,obs_horizon,desired_size,min_n_frames,s3_folder,local_path,scale = 2):
    import numpy as np
    all_videos = []
    all_labels = []
    future_cvs = {}
    tracker_dict = prune_detections(tracker_dict,min_n_frames)
    counter = 0
    validate_labels = {}
    for o,v in tracker_dict.items():
        print(counter,' objects out of', len(tracker_dict.keys()))
        all_roi_one_object = []
        all_label_one_object = []
        for f,c in v.items():
            """Decide the label If there is a lane change in that frame, check it is the appropiate object performing the LC .Do so by looking into the LC dict by the LC id"""
            lc_id = all_frames[f]
            if (lc_id != 0 and lane_changes[lc_id]['object_id'] == o):
                all_label_one_object.append(lane_changes[lc_id]['lc_class'])
                validate_labels[o]=f
                print('Object ',o, ' at frame ',f)
            else:
                all_label_one_object.append(np.array([1,0,0]))
            all_roi_one_object.append(grab_roi_from_cap(cap_path,c,f,desired_size,scale))

        #print(all_label_one_object)
        x = split_video(all_roi_one_object,obs_horizon)
        y = split_labels(all_label_one_object,0.5,obs_horizon)
        vn = save_individual_videos(s3_folder,str(o),local_path,x,obs_horizon,desired_size)
        for i in range(len(vn)):
            future_cvs[vn[i]] = y[i]

        all_videos.extend(x)
        all_labels.extend(y)
        counter+=1

    return np.stack(all_videos, axis=0), np.stack(all_labels, axis=0), future_cvs

def split_labels(labels,threshold,obs_horizon):
    final_labels = []
    amount = round(len(labels) / obs_horizon)
    start = 0
    end = obs_horizon
    for i in range(amount):
        if len(labels[start:end]) == 20:
            subset = labels[start:end]
            final_labels.append(decide_label(subset,threshold,None))
            start += 20
            end += 20
        else:
            subset = labels[start:]
            for x in range((amount * obs_horizon) - len(labels)):
                subset.append(labels[-1])
            final_labels.append(decide_label(subset,threshold,None))
    #print(final_labels)
    return final_labels

def decide_label(labels, threshold,obs_horizon):
    """Returns the final label for an entire video"""
    import numpy as np
    left = 0
    right = 0
    for i in labels:
        l = i == np.array([0, 1, 0])
        r = i == np.array([0, 0, 1])
        if l.all():
            left += 1
        elif r.all():
            right += 1


    if left / len(labels) >= threshold:
        return np.array([0, 1, 0])

    elif right / len(labels) >= threshold:
        return np.array([0, 0, 1])

    return np.array([1, 0, 0])



def prune_detections(detections,threshold):
    """Reduces the size of the detections dictionary to only keep those objects that appear in more than threshold
    frames"""
    pruned_dict = {}
    for o,v in detections.items():
        counter = 0
        for f,c in v.items():
            counter +=1
        if counter >= threshold:
            pruned_dict[o]=v
    return pruned_dict



def store_results(path,v,l,rn):
    import numpy as np
    np.save(path + '_x_' + str(rn) + '.npy' , v)
    np.save(path + '_y_' + str(rn) + '.npy',l)



def save_results_s3(v,file_type,rn):
    import boto3
    import io
    import pickle
    s3 = boto3.client('s3')
    v_array = io.BytesIO()
    pickle.dump(v,v_array)
    v_array.seek(0)
    video_name = file_type + str(rn)
    s3.upload_fileobj(v_array,'thesis-videos-dlb', video_name)



def load_results_from_s3(filename):
    import boto3
    import io
    import pickle
    array_data = io.BytesIO()
    s3_client = boto3.client('s3')
    s3_client.download_fileobj('thesis-videos-dlb',filename,array_data)
    array_data.seek(0)
    data = pickle.load(array_data)
    return data


def save_individual_videos(s3_folder,object,local_path,videos,obs_horizon,desired_size):
    """Saves the video locally and to s3
    INPUTS:
        - s3_folder: name of the s3 folder where videos should be stored
        - object: the object id that will be included in the name
        - local_path: Local path where the videos would be stored (SHOULD POINT TO THE EBS VOLUME)
        - videos: array of shape (20,220,220,3)
        - Length of the videos

    OUTPUTS:
        - video_names : LIST with the video names that will be later used to build the CSV file that contains path and label
    """
    import cv2
    import boto3
    s3 = boto3.client('s3')
    video_names = []
    fps = 20
    for idx,video in enumerate(videos):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
        vout = cv2.VideoWriter()
        key_name = object + '_' + str(idx) + '.mp4'
        success = vout.open(local_path + key_name, fourcc, fps, desired_size, True)
        for frame in range(obs_horizon):
            vout.write(video[frame])
        vout.release()
        cv2.destroyAllWindows()
        s3_key = s3_folder + '/' + key_name
        s3.upload_file(local_path + key_name, Bucket='thesis-videos-dlb', Key= s3_key)
        video_names.append(s3_key)

    return video_names







if __name__ == '__main__':
    import cv2
    import numpy as np
    import boto3
    import pandas as pd
    all_frames, lane_changes = read_lane_changes(args.lp, args.cp, args.oh, 0)
    detections = create_tracker_dictionary(args.dp)
    v,l,vl = create_X_and_Y(args.cp,all_frames,lane_changes,detections,args.oh,args.ds,args.min,args.s3,args.local,args.s)
    if args.op:
        store_results(args.op,v,l,args.rn)
    if args.ns:
        save_results_s3(v,'video',args.rn)
        save_results_s3(l, 'label', args.rn)

    s3 = boto3.client('s3')
    df = pd.DataFrame(vl.items(), columns=['path', 'label'])
    df.to_csv(args.local + args.s3 + '.csv', sep=',')
    s3.upload_file(args.local + args.s3 + '.csv', Bucket='thesis-videos-dlb', Key='csv/' + args.s3)





