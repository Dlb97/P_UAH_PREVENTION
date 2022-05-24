
from functions import *

class LoadingTest:

    def test_get_csv(self):
        filename = 'v10'
        file = get_csv(filename)
        assert type(file.columns) == list



    def test_get_video(self):
        video_name = 'v1/2_0.mp4'
        video = get_video(video_name)
        s, img = video.read()
        assert (s==True)

