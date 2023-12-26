import os, glob, json
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from decord import VideoReader, cpu, gpu 
import cv2
from camera_related_processes import grab_markers, get_median_contour, get_marker_size_px, px_to_dva

# load the data sheet if it is already existing
folder_list = glob.glob('/Users/jyeon/Documents/GitHub/pupillabs_imu/data/2023*')
data_sheet = '/Users/jyeon/Documents/GitHub/pupillabs_imu/rotated_angle.csv'
if os.path.exists(data_sheet):
    rotated_angle = pd.read_csv(data_sheet)

    # remove folder names in the list that are already processed
    folder_names = rotated_angle['folder name'].to_list()
    filtered_folder_list = [path for path in folder_list if os.path.basename(path) not in folder_names]
    folder_list = filtered_folder_list
else:
    rotated_angle = pd.DataFrame(columns = ['folder name', 'distance_markers (cm)', 'distance_wall_measured (cm)', 
                                        'distance_wall_computed (cm)', 'direction', 'session', 
                                        'start frame', 'end frame', 'rotated (deg)'])
    
# for all folders, compute the angle the eyetracker rotated based on the video 
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
    dist_markers = int(params[:params.find('cm')])
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
    center_marker_pos = center_markers[['center x', 'center y']].to_numpy()
    difference = np.linalg.norm(center_marker_pos[1:] - center_marker_pos[:-1], axis=1)
    start_frame_id = center_markers['frame'].iloc[np.where(difference>=3)[0][0]]
    end_frame_id = center_markers['frame'].iloc[np.where(difference>=3)[0][-1]]
    
    # get the final marker's median contour 
    end_marker_list = center_markers[center_markers['frame']>end_frame_id]['marker'].to_list()
    threshold = 0.6
    contour = get_median_contour(end_marker_list, threshold)
    
    # convert marker size to pixels
    marker_size_cm = (7, 6) # in cm, width and height
    
    # get marker size in pixels (width, height)
    marker_size_pixel = get_marker_size_px(contour)

    # calculate the distance of the camera from the wall
    fx, fy = cam_to_img[0,0], cam_to_img[1,1]
    dx, dy = fx*(marker_size_cm[0]/marker_size_pixel[0]), fy*(marker_size_cm[1]/marker_size_pixel[1])
    dist_cam_to_wall = np.round(np.average([dx,dy]),3)
    angle_btwn_markers = np.degrees(np.arctan(dist_markers/dist_cam_to_wall))

    # find the median location of the initial and the final markers
    start_marker_pos = center_markers[center_markers['frame']<start_frame_id][['center x', 'center y']].median().to_numpy()
    end_marker_pos = center_markers[center_markers['frame']>end_frame_id][['center x', 'center y']].median().to_numpy()
    
    # compute offset between markers and the center and convert it to visual angle
    # the marker positions are already retrieved from undistorted images
    dva_start = px_to_dva(start_marker_pos[0], start_marker_pos[1], cam_to_img, distortion, distorted = False)  # (azimuth, elevation)
    dva_end = px_to_dva(end_marker_pos[0], end_marker_pos[1], cam_to_img, distortion, distorted = False)  

    # combine and get the final angle
    angle_rotated = np.round(dva_start[0] + angle_btwn_markers - dva_end[0],2)
    
    # save the information in the data sheet
    new_row = pd.DataFrame({
        'folder name': [os.path.basename(folder)], 
        'distance_markers (cm)': dist_markers, 
        'distance_wall_measured (cm)': dist_wall, 
        'distance_wall_computed (cm)': dist_cam_to_wall, 
        'direction': [direction], 
        'session': session, 
        'start frame': start_frame_id,
        'end frame': end_frame_id,
        'rotated (deg)': angle_rotated
    })

    rotated_angle = pd.concat([rotated_angle, new_row], ignore_index=True)

# save the data sheet
rotated_angle.to_csv(data_sheet, index=False)


    