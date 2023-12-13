import numpy as np
import pandas as pd
import os, json, glob, math
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
import seaborn as sns

data_path = '/Users/jyeon/Documents/GitHub/pupillabs_imu/data'
file_list = glob.glob(os.path.join(data_path, '2023-08*'))
figure_path = '/Users/jyeon/Documents/GitHub/pupillabs_imu/imu figures'
imu_data = pd.DataFrame(columns = ['dist_wall', 'dist_origin', 'direction',
                                   'session', 'timestamp [sec]',
                                   'start time', 'end time',
                                    'gyro x [deg/s]', 'gyro y [deg/s]', 'gyro z [deg/s]',
                                    'pitch [deg]', 'yaw [deg]', 'roll [deg]'])
plt.ioff()
for f in range(len(file_list)):
    folder = os.path.join(data_path, file_list[f])
    info_path = os.path.join(folder, 'info.json')
    
    # find distance of the camera from the origin and the wall
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

    # load IMU data
    imu = pd.read_csv(os.path.join(folder, 'imu.csv'))
    events = pd.read_csv(os.path.join(folder, 'events.csv'))

    # if IMU data is missing, print it and move on to the next file
    if not np.any(imu):
        print(f'IMU data is missing for {config}')
        continue

    # organize new rows 
    times = imu['timestamp [ns]'].to_numpy()
    start_time = (events['timestamp [ns]'].loc[events['name']=='start'] - times[0])/1e9
    end_time = (events['timestamp [ns]'].loc[events['name']=='end'] - times[0])/1e9
    times = (times - times[0])/1e9
    
    new_rows = pd.DataFrame({
        'dist_wall': np.tile(np.array(dist_wall),len(imu)),
        'dist_origin': np.tile(np.array(dist_origin),len(imu)), 
        'direction': np.tile(direction,len(imu)),
        'session': np.tile(np.array(session),len(imu)),
        'timestamp [sec]': times,
        'start time': np.tile(start_time,len(imu)),
        'end time': np.tile(end_time,len(imu)),
        'gyro x [deg/s]': imu['gyro x [deg/s]'],
        'gyro y [deg/s]': imu['gyro y [deg/s]'],
        'gyro z [deg/s]': imu['gyro z [deg/s]'],
        'pitch [deg]': imu['pitch [deg]'],
        'yaw [deg]': imu['yaw [deg]'],
        'roll [deg]': imu['roll [deg]']
    })

    imu_data = pd.concat([imu_data, new_rows], ignore_index=True)

colors = sns.color_palette('Set1')
for direction in imu_data['direction'].unique():
    current_data = imu_data.loc[imu_data['direction']==direction]
    distances_from_wall = np.sort(current_data['dist_wall'].unique())
    distances_from_origin = np.sort(current_data['dist_origin'].unique())
    nCol = len(distances_from_wall)
    nRow = len(distances_from_origin)

    # set figures
    fig1, ax1 = plt.subplots(nRow, nCol, figsize=(12,10))    # gyro values traveled
    fig1.suptitle(f'Gyroscope value traveled. Moved direction: {direction}')
    fig1_name = os.path.join(figure_path, f'Gyro_traveled_{direction}.png')
    fig2, ax2 = plt.subplots(nRow, nCol, figsize=(12,10))    # raw Euler values 
    fig2.suptitle(f'Euler values, raw. Moved direction: {direction}')
    fig2_name = os.path.join(figure_path, f'Raw_Euler_{direction}.png')
    fig3, ax3 = plt.subplots(nRow, nCol, figsize=(12,10))    # normalized Euler values
    fig3.suptitle(f'Normalized Euler values. Moved direction: {direction}')
    fig3_name = os.path.join(figure_path, f'Normalized_Euler_{direction}.png')

    for dist_wall in distances_from_wall:
        col = np.where(distances_from_wall==dist_wall)[0][0]
        for dist_origin in np.sort(current_data['dist_origin'].loc[current_data['dist_wall']==dist_wall].unique()):
            row = np.where(distances_from_origin==dist_origin)[0][0]
            
            gyro_traveled = np.array([])
            euler_traveled = np.array([])
            for session in np.sort(current_data['session'].loc[(current_data['dist_wall']==dist_wall) & (current_data['dist_origin']==dist_origin)].unique()):
                session_data = current_data.loc[(current_data['dist_wall']==dist_wall) & 
                                        (current_data['dist_origin']==dist_origin) & 
                                        (current_data['session']==session)]
                start_time = session_data['start time'].unique()[0]
                end_time = session_data['end time'].unique()[0]

                # gyro values traveled  
                trimmed_data = session_data[['timestamp [sec]', 'gyro x [deg/s]', 'gyro y [deg/s]', 'gyro z [deg/s]']].loc[(session_data['timestamp [sec]']>=start_time) & (session_data['timestamp [sec]']<=end_time)]
                times = trimmed_data['timestamp [sec]'] - trimmed_data['timestamp [sec]'].iloc[0]
                x_traveled = cumtrapz(trimmed_data['gyro x [deg/s]'], times, initial=0)     
                y_traveled = cumtrapz(trimmed_data['gyro y [deg/s]'], times, initial=0)     
                z_traveled = cumtrapz(trimmed_data['gyro z [deg/s]'], times, initial=0)     

                ax1[row, col].plot(times, x_traveled, color = colors[0])    # up/down
                ax1[row, col].plot(times, z_traveled, color = colors[1])    # left/right
                ax1[row, col].plot(times, y_traveled, color = colors[2])    # roll
                if direction == 'up' or direction == 'down':
                    gyro_traveled = np.append(gyro_traveled, max(x_traveled))
                elif direction == 'left' or direction == 'right':
                    gyro_traveled = np.append(gyro_traveled, max(z_traveled))

                # Euler values - Raw
                times = session_data['timestamp [sec]'] - session_data['timestamp [sec]'].iloc[0]
                ax2[row, col].plot(times, session_data['yaw [deg]'], color = colors[0])     # up/down
                ax2[row, col].plot(times, session_data['pitch [deg]'], color = colors[1])   # left/right
                ax2[row, col].plot(times, session_data['roll [deg]'], color = colors[2])    # roll

                # Euler values - Normalized 
                trimmed_data = session_data[['timestamp [sec]', 'pitch [deg]', 'yaw [deg]', 'roll [deg]']].loc[(session_data['timestamp [sec]']>=start_time) & (session_data['timestamp [sec]']<=end_time)]
                times = trimmed_data['timestamp [sec]'] - trimmed_data['timestamp [sec]'].iloc[0]
                yaw = trimmed_data['yaw [deg]'] - trimmed_data['yaw [deg]'].iloc[0]
                pitch = trimmed_data['pitch [deg]'] - trimmed_data['pitch [deg]'].iloc[0]
                roll = trimmed_data['roll [deg]'] - trimmed_data['roll [deg]'].iloc[0]
                ax3[row, col].plot(times, yaw, color = colors[0])     # up/down
                ax3[row, col].plot(times, pitch, color = colors[1])   # left/right
                ax3[row, col].plot(times, roll, color = colors[2])    # roll
                if direction == 'up' or direction == 'down':
                    euler_traveled = np.append(euler_traveled, np.array(max(yaw)))
                elif direction == 'left' or direction == 'right':
                    euler_traveled = np.append(euler_traveled, np.array(max(pitch)))

            # set labels and everything... 
            # figure 1 - Gyro 
            axis = ax1[row, col].axis()
            ax1[row, col].set_xlim(axis[0], axis[1])
            # ax1[col, row].set_ylim(axis[2], axis[3])
            estimated_degree = np.round(np.degrees(np.arctan(dist_origin/dist_wall)),2)
            if direction == 'left' or direction == 'up':
                estimated_degree = -estimated_degree
            ax1[row, col].plot(np.array([axis[0], axis[1]]), np.array([estimated_degree]*2), color = 'r', linestyle = ':', linewidth = 2)
            if row == nRow-1:
                ax1[row, col].set_xlabel('Time [sec]')
            if col == 0:
                ax1[row, col].set_ylabel('Traveled distance [deg]')
            legend_elements = [plt.Line2D([0], [0], color=colors[0]), 
                               plt.Line2D([0], [0], color=colors[1]), 
                               plt.Line2D([0], [0], color=colors[2]), 
                            #    plt.Line2D([0], [0], color=(0.5,0.5,0.5), linestyle='dotted'), 
                               plt.Line2D([0], [0], color='r', linestyle = ':', linewidth=2)]
            legend_labels = ['Gyro - Up/Down', 'Gyro - Left/Right', 'Gyro - Roll', 
                            #  'Start/End', 
                             'Estimated traveled']
            ax1[row, col].legend(legend_elements, legend_labels, fontsize='x-small')
            avg_offset = np.round(np.mean(gyro_traveled - estimated_degree),2)
            msg = f'Wall: {distances_from_wall[col]}cm | Origin: {distances_from_origin[row]}cm \n Mean Offset: {avg_offset}deg'
            ax1[row, col].set_title(msg)

            # figure 2
            axis = ax2[row, col].axis()
            ax2[row, col].set_xlim(axis[0], axis[1])
            if row == nRow-1:
                ax2[row, col].set_xlabel('Time [sec]')
            if col == 0:
                ax2[row, col].set_ylabel('Raw Euler values [deg]')
            legend_elements = [plt.Line2D([0], [0], color=colors[0]), 
                               plt.Line2D([0], [0], color=colors[1]), 
                               plt.Line2D([0], [0], color=colors[2])]
            legend_labels = ['Yaw - Left/Right', 'Pitch - Up/Down', 'Roll']
            ax2[row, col].legend(legend_elements, legend_labels, fontsize='x-small')
            msg = f'Wall: {distances_from_wall[col]}cm | Origin: {distances_from_origin[row]}cm \n Estimated travel: {estimated_degree}deg'
            ax2[row, col].set_title(msg)

            # figure 3 
            axis = ax3[row, col].axis()
            ax3[row, col].set_xlim(axis[0], axis[1])
            ax3[row, col].plot(np.array([axis[0], axis[1]]), np.array([estimated_degree]*2), color = 'r', linestyle = ':', linewidth = 2)
            if row == nRow-1:
                ax3[row, col].set_xlabel('Time [sec]')
            if col == 0:
                ax3[row, col].set_ylabel('Normalized Euler values [deg]')
            legend_elements = [plt.Line2D([0], [0], color=colors[0]), 
                               plt.Line2D([0], [0], color=colors[1]), 
                               plt.Line2D([0], [0], color=colors[2]), 
                               plt.Line2D([0], [0], color='r', linestyle = ':', linewidth=2)]
            legend_labels = ['Yaw - Left/Right', 'Pitch - Up/Down', 'Roll', 
                             'Estimated traveled']
            ax3[row, col].legend(legend_elements, legend_labels, fontsize='x-small')
            avg_offset = np.round(np.mean(euler_traveled - estimated_degree),2)
            msg = f'Wall: {distances_from_wall[col]}cm | Origin: {distances_from_origin[row]}cm \n Mean Offset: {avg_offset}deg'
            ax3[row, col].set_title(msg)

    fig1.savefig(fig1_name, dpi=150)
    fig2.savefig(fig2_name, dpi=150)
    fig3.savefig(fig3_name, dpi=150)

    plt.close('all')
    





    