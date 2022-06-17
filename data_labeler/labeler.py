
import argparse
parser = argparse.ArgumentParser(description='Visualize resulst deep sort')
parser.add_argument('-object' ,type=int ,help='object id to follow', required= True)
args = parser.parse_args()

##

def read_object_tracking(path_to_file):
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
    return tracker



def grabnext20(cap,appereances,frame):
    frames = []
    counter = 0
    for f,c in appereances.items():
        if counter < 20:
            if f > frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f )
                success, img = cap.read()
                frames.append(img[c[1]:c[1] + c[3],c[0] : c[0] + c[2]])
                counter+=1
        else:
            break
    if counter != 20:
        for i in range(20 - counter):
            frames.append(frames[-1])
    return frames


def visualize_results(cap ,object_id ,tracker):
    """Shows the detections for a specific object"""
    import cv2
    import time
    appereances = tracker[object_id]
    for k ,v in appereances.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        success, img = cap.read()
        cv2.rectangle(img ,(v[0] ,v[1]) ,(v[0 ] +v[2] ,v[1 ] +v[3]) ,(0 ,255 ,0) ,2)
        cv2.imshow('Results ' +str(object_id) ,img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()



def visualize_all_results(cap ,tracker):
    """Shows ALL the detections"""
    import cv2
    import time
    for o in tracker.keys():
        appereances = tracker[o]
        for k ,v in appereances.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            success, img = cap.read()
            cv2.rectangle(img ,(v[0] ,v[1]) ,(v[0 ] +v[2] ,v[1 ] +v[3]) ,(0 ,255 ,0) ,2)
            cv2.imshow('Results' ,img)
            cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(10)

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
    detections = read_object_tracking('/Users/david/workspace/thesis/results_processed_videos/video_1/complete_v1.txt')
    print(detections.keys())
    """
    for i in detections.keys():
        visualize_results(cap,i,detections)
    #visualize_results(cap,args.object,detections)
        time.sleep(1)
    """
    visualize_all_results(cap,detections)
