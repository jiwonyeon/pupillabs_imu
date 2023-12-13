import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader, cpu, gpu 
import cv2
from camera_related_processes import grab_markers


def main(folder):
    # get the video information 
    info_path = os.path.join(folder, 'info.json')
    config = None
    with open(info_path, 'r') as file:
        info = json.load(file)
        key = list(info['template_data']['data'].keys())
        config = info['template_data']['data'][key[0]][0]

    # find degrees the glasses moved
    params = config
    dist_wall = int(config[:params.find('cm')])
    params = params[params.find('cm')+3:]
    dist_origin = int(params[:params.find('cm')])
    direction = params[params.find('cm')+3:-2]
    session = int(config[-1])

    # load video
    video_path = glob.glob(os.path.join(folder, '*.mp4'))[0]
    video = VideoReader(video_path, ctx=cpu(0))

    # grab markers on the wall 
    markers = grab_markers(video)
    




    pass


    


if __name__ == '__main__':
    folder_list = glob.glob('/Users/jyeon/Documents/GitHub/pupillabs_imu/data/2023*')
    for f in range(len(folder_list)):
        folder = folder_list[f]
        main(folder)

    