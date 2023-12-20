import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader, cpu, gpu 
import cv2
from camera_related_processes import grab_markers, find_median_contour

folder_list = glob.glob('/Users/jyeon/Documents/GitHub/pupillabs_imu/data/2023*')

for f in range(len(folder_list)):
    folder = folder_list[f]

    # get the video information 
    info_path = os.path.join(folder, 'info.json')
    config = None
    with open(info_path, 'r') as file:
        info = json.load(file)
        key = list(info['template_data']['data'].keys())
        config = info['template_data']['data'][key[0]][0]

    # pull out the session info
    params = config
    dist_wall = int(config[:params.find('cm')])
    params = params[params.find('cm')+3:]
    dist_origin = int(params[:params.find('cm')])
    direction = params[params.find('cm')+3:-2]
    session = int(config[-1])

    # load camera intrinsics
    cam_to_img = np.array(json.load(open(folder + '/scene_camera.json'))['camera_matrix'])
    # load distortion coefficients
    distortion = np.array(json.load(open(folder + '/scene_camera.json'))['distortion_coefficients'])

    # load video
    video_path = glob.glob(os.path.join(folder, '*.mp4'))[0]
    video = VideoReader(video_path, ctx=cpu(0))

    # grab markers closest to the center of frames
    center_markers = grab_markers(video, cam_to_img, distortion)
    center_markers = center_markers.dropna()
    
    # find the start and the end frames
    center_pos = center_markers[['center x', 'center y']].to_numpy()
    difference = np.linalg.norm(center_pos[1:] - center_pos[:-1], axis=1)
    start_frame_id = center_markers['frame'].iloc[np.where(difference>=3)[0][0]]
    end_frame_id = center_markers['frame'].iloc[np.where(difference>=3)[0][-1]]
    
    # find the median location of the initial and the final markers
    start_pos = center_markers[center_markers['frame']<start_frame_id][['center x', 'center y']].median().to_numpy()
    end_pos = center_markers[center_markers['frame']>end_frame_id][['center x', 'center y']].median().to_numpy()

    # get the final marker's median contour to convert pixel to cm 
    end_marker_contour = np.squeeze(np.concatenate(center_markers[center_markers['frame']>end_frame_id]['marker'].to_list()))
    median_contours = find_median_contour(end_marker_contour)
    

    # calculate pixels to cm using a known marker size
    ### this calculation is based on the size of the origin (end) marker
    end_marker_size = (7, 6.5) # in cm, width and height
    



    