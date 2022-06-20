my_path="/mounted_ebs/input_data"

for folder in v10 v11 v2 v3 v4 v5 v6 v7 v8 v9
do
         files_path=$my_path/$folder
 	 echo $files_path
         $(sudo python3 /mounted_ebs/process_only_LC.py -cp $files_path/video_camera1.mp4 -dp $files_path/processed_data/detection_camera1/detections_tracked.txt  \
         -lp $files_path/processed_data/detection_camera1/lane_changes.txt -tte 20 -o ONLY_LC -sp /mounted_ebs/positive_videos/ )

done
