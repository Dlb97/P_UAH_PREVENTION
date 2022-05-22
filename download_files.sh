mkdir $1
cd $1
wget https://prevention-dataset.uah.es/static/RECORD$2/DRIVE$3/video_camera1.mp4 --no-check-certificate
wget https://prevention-dataset.uah.es/static/RECORD$2/DRIVE$3/processed_data.zip --no-check-certificate
unzip processed_data.zip
rm processed_data.zip
