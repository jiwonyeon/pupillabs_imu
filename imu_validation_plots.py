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
fig_path = './imu figures'

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

    # set figures
    n_col = 2
    n_row = int(np.ceil(len(dist_params)/2))
    row, col = 0, 0
    fig_gyro, ax1 = plt.subplots(n_row, n_col, figsize = (8,8))
    fig_euler, ax2 = plt.subplots(n_row, n_col, figsize = (8,8))
    fig_gyro.subplots_adjust(wspace=0.2, hspace=0.3)
    fig_euler.subplots_adjust(wspace=0.2, hspace=0.3)

    for dist_wall, dist_markers in dist_params:
        # find the folders that corresponds to the parameters
        list_same_condition = short_list.loc[(short_list['distance_wall_measured (cm)']==dist_wall) &
                                 (short_list['distance_markers (cm)']==dist_markers)]

        # create an empty variables to save imu_changes
        X_gyro, Y_gyro, Z_gyro = np.zeros((0,500)), np.zeros((0,500)), np.zeros((0,500))
        Pitch, Yaw, Roll = np.zeros((0,500)), np.zeros((0,500)), np.zeros((0,500))
        for id, folder in enumerate(list_same_condition['folder name']):
            # retrieve the IMU data
            imu = pd.read_csv(os.path.join(data_path, folder, 'imu.csv'))

            # if a folder doesn't have imu, skip the folder
            if imu.empty:
                print(f"Folder {folder} has emtpy imu file")
                continue

            video_timestamp= pd.read_csv(os.path.join(data_path, folder, 'world_timestamps.csv'))
            start_time = video_timestamp['timestamp [ns]'].loc[list_same_condition['start frame'].iloc[id]]
            end_time = video_timestamp['timestamp [ns]'].loc[list_same_condition['end frame'].iloc[id]]
            imu = imu.loc[(imu['timestamp [ns]']>=start_time) & (imu['timestamp [ns]']<=end_time)]

            # compute times
            times = imu['timestamp [ns]'].to_numpy()/1e9
            times = times-times[0]

            # compute cumulative trapezoidal integration for gyro
            # x = up/down (pitch), y = roll (roll), z = left/right (yaw)
            gyro_vals = {
                'x': cumtrapz(imu['gyro x [deg/s]'], times, initial=0),
                'y': cumtrapz(imu['gyro y [deg/s]'], times, initial=0),
                'z': cumtrapz(imu['gyro z [deg/s]'], times, initial=0)}
            
            # match pitch, yaw, and roll to zeros in the beginning
            euler_vals = {
                'pitch': imu['pitch [deg]'].to_numpy() - np.mean(imu['pitch [deg]'].iloc[:20]),
                'yaw': imu['yaw [deg]'].to_numpy() - np.mean(imu['yaw [deg]'].iloc[:20]),
                'roll': imu['roll [deg]'].to_numpy() - np.mean(imu['roll [deg]'].iloc[:20])}
            
            # keep the number of samples to 500 
            new_times = np.linspace(times[0], times[-1], 500)
            gyro = {
                'x': np.interp(new_times, times, gyro_vals['x']), 
                'y': np.interp(new_times, times, gyro_vals['y']), 
                'z': np.interp(new_times, times, gyro_vals['z'])}
            
            euler = {
                'pitch': np.interp(new_times, times, euler_vals['pitch']), 
                'yaw': np.interp(new_times, times, euler_vals['yaw']), 
                'roll': np.interp(new_times, times, euler_vals['roll'])}
            
            # save each trial's result and plot it
            X_gyro, Z_gyro, Y_gyro = np.vstack((X_gyro, gyro['x'])), np.vstack((Z_gyro, gyro['z'])), np.vstack((Y_gyro, gyro['z']))
            Pitch, Yaw, Roll = np.vstack((Pitch, euler['pitch'])), np.vstack((Yaw, euler['yaw'])), np.vstack((Roll, euler['roll']))

            # plot gyro
            times = np.linspace(0, 1, 500)
            ax1[row, col].plot(times, gyro['x'], color = colors[0], linewidth = 1, alpha = 0.6)
            ax1[row, col].plot(times, gyro['z'], color = colors[1], linewidth = 1, alpha = 0.6)
            ax1[row, col].plot(times, gyro['y'], color = colors[2], linewidth = 1, alpha = 0.6)
            ax2[row, col].plot(times, euler['pitch'], color = colors[0], linewidth=1, alpha = 0.6)
            ax2[row, col].plot(times, euler['yaw'], color = colors[1], linewidth=1, alpha = 0.6)
            ax2[row, col].plot(times, euler['roll'], color = colors[2], linewidth=1, alpha = 0.6)
            
        # plot average track of the movement 
        gyro_avg = np.array([np.mean(X_gyro, axis=0), np.mean(Z_gyro, axis=0), np.mean(Y_gyro, axis=0)])
        euler_avg = np.array([np.mean(Pitch, axis=0), np.mean(Yaw, axis=0), np.mean(Roll, axis=0)])
        for axis_id in range(3):
            ax1[row, col].plot(times, gyro_avg[axis_id], color = colors[axis_id], linewidth = 1.5)
            ax2[row, col].plot(times, euler_avg[axis_id], color = colors[axis_id], linewidth = 1.5)
        
        # plot the computed angle traveled
        avg_computed_angle = list_same_condition['rotated (deg)'].mean()
        if direction in ('left', 'up'):
            avg_computed_angle = -avg_computed_angle
        ax1[row, col].plot(times, np.tile(avg_computed_angle, len(times)), '--r', linewidth = 1.5)
        ax2[row, col].plot(times, np.tile(avg_computed_angle, len(times)), '--r', linewidth = 1.5)
        
        # add the legend 
        legend_elements = [plt.Line2D([0], [0], color=colors[0]),                            
                    plt.Line2D([0], [0], color=colors[1]), 
                    plt.Line2D([0], [0], color=colors[2]),
                    plt.Line2D([0], [0], color='r', linestyle = '--', linewidth=1.5)]
        legend_labels = ['Up/Down', 'Left/Right', 'Roll', 'Estimated rotated']
        ax1[row, col].legend(legend_elements, legend_labels, fontsize='x-small')
        ax1[row, col].set_xlim([0, 1])
        if col == 0:
            ax1[row, col].set_ylabel('Rotated [deg]')
        if row+1 == n_row:
            ax1[row, col].set_xlabel('Normalized time')
        
        legend_labels = ['Up/Down [Pitch]', 'Left/Right [Yaw]', 'Roll [Roll]', 'Estimated rotated']
        ax2[row, col].legend(legend_elements, legend_labels, fontsize='x-small')
        ax2[row, col].set_xlim([0, 1])
        if col == 0:
            ax2[row, col].set_ylabel('Rotated [deg]')    
        if row+1 == n_row:
            ax2[row, col].set_xlabel('Normalized time')

        # add the title            
        avg_offset_gyro = np.sort(np.max(np.abs(np.vstack([X_gyro, Y_gyro, Z_gyro])), axis=1))[-len(list_same_condition):] - np.abs(avg_computed_angle)
        avg_offset_euler = np.sort(np.max(np.abs(np.vstack([Pitch, Yaw, Roll])), axis=1))[-len(list_same_condition):] - np.abs(avg_computed_angle)
        ax1[row, col].set_title(f'camera dist: {dist_wall}, marker dist: {dist_markers}, \noffset: {np.round(np.mean(np.abs(avg_offset_gyro)),2)}', 
                                fontsize=10)
        ax2[row, col].set_title(f'camera dist: {dist_wall}, marker dist: {dist_markers}, \noffset: {np.round(np.mean(np.abs(avg_offset_euler)),2)}', 
                                fontsize=10)

        # update row and col
        if col+1 == n_col:
            col = 0 
            row += 1
        else:
            col += 1

    # add supertitle
    fig_gyro.suptitle(f'{direction.capitalize()}')
    fig_euler.suptitle(f'{direction.capitalize()}')

    # save figures
    fig_name = os.path.join(fig_path, 'gyro_video_analysis_'+direction+'.png')
    fig_gyro.savefig(fig_name, dpi=150)
    fig_name = os.path.join(fig_path, 'euler_video_analysis_'+direction+'.png')
    fig_euler.savefig(fig_name, dpi=150)