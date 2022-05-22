# P_UAH_PREVENTION
Preprocessing for the UAH Prevention Dataset (https://prevention-dataset.uah.es). The final goal is to have short individual videos for each object detected that can be later feed to a ML model to predict Lane Changes. The repo is composed of the following files:


* data_join.py: File that joins the detections_tracked.txt , the lane_changes.txt and the video for one recording. The outputs that will be produced after running the file are the following :
 
  1 Smaller videos of a certain number of frames for each object detected. So if we have an object (let's say car with id 6) that appears in a total of 200 frames, we will make 10 shorter videos of 20 frames each (Configurable parameters are availabe to specify the length). The videos in mp4 format can be stored locally or in AWS S3.
  
  2 A 5D numpy array that contain all the individual videos [N 20 224 224 3]  + labels for the videos that contain a future LC (also as an array and taking into account the obs horizon mentioned in the paper)  
  
  3 A CSV file with two columns (Path, Label). The idea is to use it as a dataloader to not have all the videos produced in 1 loaded in memory.
  
  
  
  
* download_files: Creates a directory and download the files (txt data about detections and lane changes + video). Example to use it :

     
