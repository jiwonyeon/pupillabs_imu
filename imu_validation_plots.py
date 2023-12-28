# validate whether the IMU data shows correctly the eyetracker device rotated
# run detect_movement_from_video.py first to generate a summary result of the device rotated

import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.signal import resample
from decord import VideoReader, cpu, gpu 
import seaborn as sns

# load the summary result
rotated_angle = pd.read_csv('rotated_angle.csv')
data_path = '/Users/jyeon/Documents/GitHub/pupillabs_imu/data'

# set colors 
colors = sns.color_palette('Set2')

# compute the data according to the direction/distances of the eyetracker device
directions = ('right', 'left', 'up', 'down')
for direction in directions:
    # filter the datalist
    short_list = rotated_angle.loc[rotated_angle['direction'] == direction]
    dist_params = short_list[['distance_wall_measured (cm)', 'distance_markers (cm)']].drop_duplicates().to_numpy()
    
    # sort the order from closest to farthest
    sort_index = np.lexsort((dist_params[:, 1], dist_params[:, 0]))
    dist_params = dist_params[sort_index,:]

    # set number of rows and columns for the figure
    n_col = 2
    n_row = np.ceil(dist_params/2)
    fig_num = 0

    for dist_wall, dist_markers in dist_params:
        # find the folders that corresponds to the parameters
        list_same_condition = short_list.loc[(short_list['distance_wall_measured (cm)']==dist_wall) &
                                 (short_list['distance_markers (cm)']==dist_markers)]

        # create an empty variables to save imu_changes
        X_gyro, Y_gyro, Z_gyro = np.zeros((0,500)), np.zeros((0,500)), np.zeros((0,500))
        Pitch, Yaw, Roll = np.zeros((0,500)), np.zeros((0,500)), np.zeros((0,500))
        fig_gyro, ax1 = plt.subplots(n_row, n_col, figsize = (8,4))
        fig_euler, ax2 = plt.subplots(n_row, n_col, figsize = (8,4))

        ax1[fig_num].set_title(f'camera dist: {dist_wall}, marker dist: {dist_markers}')
        ax2[fig_num].set_title(f'camera dist: {dist_wall}, marker dist: {dist_markers}')
        
        for id, folder in enumerate(list_same_condition['folder name']):
            # retrieve the IMU data
            imu = pd.read_csv(os.path.join(data_path, folder, 'imu.csv'))
            video_timestamp= pd.read_csv(os.path.join(data_path, folder, 'world_timestamps.csv'))
            start_time = video_timestamp['timestamp [ns]'].loc[list_same_condition['start frame'].iloc[id]]
            end_time = video_timestamp['timestamp [ns]'].loc[list_same_condition['end frame'].iloc[id]]
            imu = imu.loc[(imu['timestamp [ns]']>=start_time) & (imu['timestamp [ns]']<=end_time)]

            # compute times
            times = imu['timestamp [ns]'].to_numpy()/1e9
            times = times-times[0]

            # compute cumulative trapezoidal integration for gyro
            # x = up/down (pitch), y = roll (roll), z = left/right (yaw)
            gyro_vals = np.squeeze(np.array([[cumtrapz(imu['gyro x [deg/s]'], times, initial=0)], 
                         [cumtrapz(imu['gyro z [deg/s]'], times, initial=0)],
                         [cumtrapz(imu['gyro y [deg/s]'], times, initial=0)]]))

            # match pitch, yaw, and roll to zeros in the beginning
            euler_vals = np.squeeze(np.array([[imu['pitch [deg]'].to_numpy() - np.mean(imu['pitch [deg]'].iloc[:20])], 
                          [imu['yaw [deg]'].to_numpy() - np.mean(imu['yaw [deg]'].iloc[:20])],
                          [imu['roll [deg]'].to_numpy() - np.mean(imu['roll [deg]'].iloc[:20])]]))
            
            # keep the number of samples to 500 
            gyro = np.empty((3, 500))
            euler = np.empty((3, 500))
            new_times = np.linspace(times[0], times[-1], 500)
            if len(imu) != 500:
                for axis_id in range(3):
                    gyro[axis_id,:] = np.interp(new_times, times, gyro_vals[axis_id,:])
                    euler[axis_id,:] = np.interp(new_times, times, euler_vals[axis_id,:])
            
            # save each trial's result and plot it
            X_gyro, Z_gyro, Y_gyro = np.vstack((X_gyro, gyro[0,:])), np.vstack((Z_gyro, gyro[1,:])), np.vstack((Y_gyro, gyro[2,:]))
            Pitch, Yaw, Roll = np.vstack((Pitch, euler[0,:])), np.vstack((Yaw, euler[1,:])), np.vstack((Roll, euler[2,:]))
            
            # plot gyro
            times = np.linspace(0, 1, 500)
            for axis_id in range(3):
                ax1[fig_num].plot(times, gyro[axis_id,:], color = colors[axis_id], linewidth = 1, alpha = 0.6)
                ax2[fig_num].plot(times, euler[axis_id,:], color = colors[axis_id], linewidth=1, alpha = 0.6)
            
        # plot average track of the movement 
        gyro_avg = np.array([np.mean(X_gyro, axis=0), np.mean(Z_gyro, axis=0), np.mean(Y_gyro, axis=0)])
        euler_avg = np.array([np.mean(Pitch, axis=0), np.mean(Yaw, axis=0), np.mean(Roll, axis=0)])
        for axis_id in range(3):
            ax1[fig_num].plot(times, gyro_avg[axis_id], color = colors[axis_id], linewidth = 1.5)
            ax2[fig_num].plot(times, euler_avg[axis_id], color = colors[axis_id], linewidth = 1.5)
        
        # plot the computed angle traveled
        avg_computed_angle = list_same_condition['rotated (deg)'].mean()
        if direction in ('left', 'up'):
            avg_computed_angle = -avg_computed_angle
        ax1[fig_num].plot(times, np.tile(avg_computed_angle, len(times)), '--r', linewidth = 1.5)
        ax2[fig_num].plot(times, np.tile(avg_computed_angle, len(times)), '--r', linewidth = 1.5)

        legend_elements = [plt.Line2D([0], [0], color=colors[0]),                            
                            plt.Line2D([0], [0], color=colors[1]), 
                            plt.Line2D([0], [0], color=colors[2]),
                            plt.Line2D([0], [0], color='r', linestyle = '--', linewidth=1.5)]
        
        # add the legend 
        legend_labels = ['Up/Down', 'Left/Right', 'Roll', 'Estimated rotated']
        ax1[fig_num].legend(legend_elements, legend_labels, fontsize='x=small')

        legend_labels = ['Up/Down [Pitch]', 'Left/Right [Yaw]', 'Roll [Roll]', 
                        'Estimated rotated']
        ax2[fig_num].legend(legend_elements, legend_labels, fontsize='x-small')
        
