import numpy as np
import pandas as pd
import os, json, glob, math
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz

data_path = '/Users/jyeon/Documents/GitHub/pupillabs_imu/data'
file_list = glob.glob(os.path.join(data_path, '2023-08*'))
figure_path = '/Users/jyeon/Documents/GitHub/pupillabs_imu/imu figures'
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
    degrees_moved = round(math.degrees(np.arctan(dist_origin/dist_wall)),2)
    trial = int(config[-1])

    # load IMU data
    imu = pd.read_csv(os.path.join(folder, 'imu.csv'))
    events = pd.read_csv(os.path.join(folder, 'events.csv'))

    # if IMU data is missing, print it and move on to the next file
    if not np.any(imu):
        print(f'IMU data is missing for {config}')
        continue

    # cut the IMU data to between the start and the end points
    start_time = events['timestamp [ns]'].loc[events['name']=='start'].to_numpy()[0]
    end_time = events['timestamp [ns]'].loc[events['name']=='end'].to_numpy()[0]
    delete_rows = np.hstack([np.where((imu['timestamp [ns]']<start_time)==True)[0],
                            np.where((imu['timestamp [ns]']>end_time)==True)[0]])
    imu = imu.drop(delete_rows, axis=0)

    # Gyro angular velocity data
    figure_name = 'Gyro_AngularVelocity_' + config + '.png'
    fig, ax = plt.subplots()
    times = imu['timestamp [ns]'].to_numpy()
    times = (times - times[0])/1e9
    ax.plot(times, imu['gyro x [deg/s]'], label='Gyro x')
    ax.plot(times, imu['gyro y [deg/s]'], label = 'Gyro y')
    ax.plot(times, imu['gyro z [deg/s]'], label = 'Gyro z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular velocity (deg/s)')
    ax.legend()
    plt.title(f'Gyroscope angular velocity {config}')
    fig.savefig(os.path.join(figure_path, figure_name), dpi=150)

    # Gyro traveled
    figure_name = 'Gyro_Traveled_' + config + '.png'
    x_traveled = cumtrapz(imu['gyro x [deg/s]'], times, initial=0)
    y_traveled = cumtrapz(imu['gyro y [deg/s]'], times, initial=0)
    z_traveled = cumtrapz(imu['gyro z [deg/s]'], times, initial=0)
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(times, x_traveled, label = 'x')
    ax.plot(times, y_traveled, label = 'y')
    ax.plot(times, z_traveled, label = 'z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    plt.title(f'Gyro traveled {config} \nExpected traveled degree: {degrees_moved}deg')
    fig.savefig(os.path.join(figure_path, figure_name), dpi=150)

    # Rotation
    figure_name = 'Rotation_' + config + '.png'
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(times, imu['pitch [deg]'], label = 'Pitch')
    ax.plot(times, imu['yaw [deg]'], label = 'Yaw')
    ax.plot(times, imu['roll [deg]'], label = 'Roll')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    plt.title(f'Rotation data {config} \nExpected traveled degree: {degrees_moved}deg')
    fig.savefig(os.path.join(figure_path, figure_name), dpi=150)

 





    