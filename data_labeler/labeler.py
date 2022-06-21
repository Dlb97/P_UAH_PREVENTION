
import argparse
from email.mime import image
parser = argparse.ArgumentParser(description='Visualize resulst deep sort')
parser.add_argument('-cp' ,type=str,help='path to video', required= True)
parser.add_argument('-dp' ,type=str,help='path to detections', required= True)
parser.add_argument('-n' ,type=str,help='name of the video', required= True)
parser.add_argument('-t' ,type=int,help='thresold of number of observations', required= True)
parser.add_argument('-o' ,type=str,help='file where to store video path and label', required= True)
parser.add_argument('-c',type=str,help='path to checkpoint file',required=True)
parser.add_argument('-r',type=str,help='resume where we left?',required=False)
args = parser.parse_args()

"""
python3 labeler.py -cp /Users/david/workspace/thesis/extra_videos/germany.mp4 -dp /Users/david/workspace/thesis/extra_videos/germany.txt -n germany -t 80 \
-o /Users/david/workspace/thesis/thesis_repo/P_UAH_PREVENTION/data_labeler/checkpoints/germany_info \
-c /Users/david/workspace/thesis/thesis_repo/P_UAH_PREVENTION/data_labeler/checkpoints/germany_checkpoints \
-r True
"""

def read_object_tracking(path_to_file,threshold,checkpoint_file,resume):

    """Reads the results of DEEP SORT ALGORITHM txt files to create and return a dictionary withe the following format
     { object_id: { frame_id : (x,y,w,h), frame_id : (x,y,w,h) } , object_id: { frame_id : (x,y,w,h), frame_id : (x,y,w,h) } }

     args:
        - resume : path to checkpoint file
     """

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
    #Discard objects with low nº of observations and resume labeling
    if (resume == 'True'):
        last_object = get_last_object_processed(checkpoint_file)
        tracker = {k: v for k, v in tracker.items() if (len(v) > threshold and k > last_object)}
    else:
        tracker = {k: v for k, v in tracker.items() if len(v) > threshold}
    return tracker




def visualize_results(cap ,object_id ,tracker, video_name,file_name,checkpoint_file):
    """Shows the detections for a specific object"""
    import cv2
    import time
    appereances = tracker[object_id]
    for k ,v in appereances.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        success, img = cap.read()
        try:
            cv2.rectangle(img ,(v[0] ,v[1]) ,(v[0] +v[2] ,v[1] +v[3]) ,(0 ,255 ,0) ,2)
            cv2.imshow('Results ' +str(object_id) ,img)
        except cv2.error:
            print('CV2ERROR for object {} at frame {}'.format(object_id,k))
            print(v)
            print(img)
        button = cv2.waitKey(1)
        print(button)
        if (button == ord('l') or button == ord('r')):
            frames = grabnext20(cap,appereances,k)
            v_name =save_individual_video('data_labeler',video_name,k,'/Users/david/workspace/thesis/thesis_repo/P_UAH_PREVENTION/data_labeler/selected_videos/',frames,(224,224))
            save_info(file_name,v_name,str(button))
        elif button == ord('s'):
            create_checkpoint(str(object_id),checkpoint_file)
            break
        elif button == -1:
            pass
    




def visualize_all_results(cap ,tracker, video_name, file_name,checkpoint_file):
    """Shows ALL the detections"""
    import cv2
    import time
    for o in tracker.keys():
        visualize_results(cap,o,tracker,video_name, file_name,checkpoint_file)
        time.sleep(2)


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



def get_last_object_processed(path_to_checkpoint):
    """Grabs the object id of the last object processed. With this function we can stop and restart the labeling from where we left
    """
    with open(path_to_checkpoint,'r') as file:
        lines = file.read().splitlines()
        return int(lines[-1])



def create_checkpoint(object_id,checkpoint_file):
    try:
        with open(checkpoint_file, 'a') as file:
            file.write(object_id)
            file.write('\n')
    except FileNotFoundError:
        with open(checkpoint_file, 'w') as file:
            file.write(object_id)
            file.write('\n')






if __name__ == '__main__':
    import cv2
    cap = cv2.VideoCapture(args.cp)
    detections = read_object_tracking(args.dp,args.t,args.c,args.r)
    visualize_all_results(cap,detections,args.n,args.o,args.c)
