my_path="/mounted_ebs/input_data"

for folder in v10 v11 v2 v3 v4 v5 v6 v7 v8 v9
do
         files_path=$my_path/$folder
 	 echo $files_path
         $(sudo python3 /mounted_ebs/data_join.py -cp $files_path/video_camera1.mp4 -dp $files_path/processed_data/detection_camera1/detections_tracked.txt  \
         -lp $files_path/processed_data/detection_camera1/lane_changes.txt -oh 20 -ds 224 224 -min 140 -s 3 -rn $folder -op $files_path/arrays/ \
         -s3 $folder -local $my_path/individual_videos/ )

done
