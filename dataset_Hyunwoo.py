import numpy as np
import pandas as pd
import os

data = pd.read_csv('rotated_angle.csv')

hw_data = pd.DataFrame(columns = ['folder name', 'section id', 'recording id', 
                                  'dist marker (cm)', 'dist wall measured (cm)', 'dist wall computed (cm)', 
                                  'direction', 'session', 'start frame', 'end frame',
                                  'rotated computed (deg)', 
                                  'frame id', 
                                  'timestamp [ns]', 'gyro x [deg/s]', 'gyro y [deg/s]', 'gyro z [deg/s]', 
                                  'acceleration x [G]', 'acceleration y [G]', 'acceleration z [G]',
                                  'pitch [deg]', 'yaw [deg]', 'roll [deg]', 
                                  'quaternion w', 'quaternion x', 'quaternion y', 'quaternion z'
                                  ])

for id, folder in enumerate(data['folder name']):
    thisrow = data.loc[data['folder name']==folder]
    imu = pd.read_csv(os.path.join('./data', folder, 'imu.csv'))

    new_names = ['folder name', 'dist marker (cm)', 'dist wall measured (cm)', 'dist wall computed (cm)', 
                 'direction', 'session', 'start frame', 'end frame', 'rotated computed (deg)']
    old_names = ['folder name', 'distance_markers (cm)', 'distance_wall_measured (cm)', 
                 'distance_wall_computed (cm)', 'direction', 'session', 'start frame', 'end frame', 'rotated (deg)']
    new_imu = imu
    for new_name, old_name in zip(new_names, old_names):
        new_imu = new_imu.assign(**{new_name: thisrow[old_name].iloc[0]})

    new_imu['frame id'] = new_imu.index
    hw_data = pd.concat([hw_data, new_imu], ignore_index=True)

hw_data.to_csv('all_imu_data.csv')