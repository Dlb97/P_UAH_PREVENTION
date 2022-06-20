
import argparse
parser = argparse.ArgumentParser(description='Visualize resulst deep sort')
parser.add_argument('-object' ,type=int ,help='object id to follow', required= True)
args = parser.parse_args()

##

def read_object_tracking(path_to_file,threshold):
    """Reads the results of DEEP SORT ALGORITHM txt files to create and return a dictionary withe the following format
     { object_id: { frame_id : (x,y,w,h), frame_id : (x,y,w,h) } , object_id: { frame_id : (x,y,w,h), frame_id : (x,y,w,h) } } """

    tracker = {}
    with open(path_to_file) as file:
        for line in file.readlines():
            line = line = line.rstrip('\n').split(' ')
            object = int(line[1])
            frame_id = int(line[0])
            x = int(line[2])
            y = int(line[3])
            w = int(line[4])
            h =  int(line[5])
            if object not in tracker.keys():
                tracker[object] = {frame_id :(x ,y ,w ,h)}
            else:
                tracker[object][frame_id] = (x ,y ,w ,h)
    #Discard objects with low nº of observations
    tracker = {k: v for k, v in tracker.items() if len(v) > threshold}
    return tracker




def visualize_results(cap ,object_id ,tracker, video_name,file_name):
    """Shows the detections for a specific object"""
    import cv2
    import time
    appereances = tracker[object_id]
    for k ,v in appereances.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        success, img = cap.read()
        try:
            cv2.rectangle(img ,(v[0] ,v[1]) ,(v[0 ] +v[2] ,v[1 ] +v[3]) ,(0 ,255 ,0) ,2)
            cv2.imshow('Results ' +str(object_id) ,img)
        except cv2.error:
            pass
        button = cv2.waitKey(3)
        if (button == ord('l') or button == ord('r')):
            frames = grabnext20(cap,appereances,k)
            v_name =save_individual_video('data_labeler',video_name,k,'./data_labeler/selected_videos/',frames,(224,224))
            save_info(file_name,video_name,button)
    cap.release()
    cv2.destroyAllWindows()



def visualize_all_results(cap ,tracker, video_name, file_name):
    """Shows ALL the detections"""
    import cv2
    import time
    for o in tracker.keys():
        visualize_results(cap,o,tracker,video_name, file_name)


def grabnext20(cap,appereances,frame):
    """Given the detections for a specific object at a specific frame. Grab the next 20 frames where that object
    appears"""
    frames = []
    counter = 0
    for f,c in appereances.items():
        if counter < 20:
            if f > frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f )
                success, img = cap.read()
                img = img[c[1]:c[1] + c[3],c[0] : c[0] + c[2]]
                frames.append(pad_image((224,224),img))
                counter+=1
        else:
            break
    if counter != 20:
        for i in range(20 - counter):
            frames.append(frames[-1])
    return frames





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


def save_individual_video(s3_folder,video_name,frame,local_path,video,desired_size):
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




def save_info(file_name,video_name,label):
    """Writes the s3 path and the label in a txt file for later training purposes"""
    try:
        with open(file_name,'a') as file:
            file.write(video_name)
            file.write(',')
            file.write(label)
            file.write('\n')
    except FileNotFoundError:
        with open(file_name,'w') as file:
            file.write(video_name)
            file.write(',')
            file.write(label)
            file.write('\n')


##

detections = read_object_tracking('/Users/david/workspace/thesis/results_processed_videos/video_1/complete_v1.txt')
##
if __name__ == '__main__':
    import cv2
    import time

    # cap = cv2.VideoCapture('/Users/david/workspace/thesis/results_processed_videos/video_1/complete_v1.mp4')
    cap = cv2.VideoCapture('/Users/david/workspace/thesis/PREVENTION-DATASET/video_camera1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES ,0)
    success, img = cap.read()
    detections = read_object_tracking('/Users/david/workspace/thesis/results_processed_videos/video_1/complete_v1.txt',100)
    print(detections.keys())
    """
    for i in detections.keys():
        visualize_results(cap,i,detections)
    #visualize_results(cap,args.object,detections)
        time.sleep(1)
    """
    visualize_all_results(cap,detections)
