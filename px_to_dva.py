import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import json
import pandas as pd
import argparse

def dva_to_px(elevation, azimuth, camera_matrix, distortion_coeffs):
    """
    Convert from degrees of visual angle to pixels

    Parameters
    ----------
    elevation : float
        Elevation in degrees
    azimuth : float
        Azimuth in degrees
    camera_matrix : numpy.ndarray
        Camera matrix
    distortion_coeffs : numpy.ndarray
        Distortion coefficients

    Returns
    -------
    numpy.ndarray
        2D coordinates in pixels
    """
    # Rotate the point about the x and y axes
    rot_x = R.from_euler('x', elevation, degrees=True)
    rot_y = R.from_euler('y', azimuth, degrees=True)
    
    # Compose the rotations
    rot_composed = rot_y * rot_x
    
    # Apply the composed rotation to the point
    img_center = np.array([0, 0, 1])
    rotated_point = rot_composed.apply(img_center)
    
    # Project to 2D
    distorted_point = cv2.projectPoints(rotated_point.reshape(1, 1, 3), np.zeros((3,1)), np.zeros((3,1)), camera_matrix, distortion_coeffs)[0]
    
    return distorted_point[0][0]

def px_to_dva(x_px, y_px, camera_matrix, distortion_coeffs):
    """
    Convert from pixels to degrees of visual angle (DVA)

    Parameters
    ----------
    pixel_coords : numpy.ndarray
        2D coordinates in pixels
    camera_matrix : numpy.ndarray
        Camera matrix
    distortion_coeffs : numpy.ndarray
        Distortion coefficients

    Returns
    -------
    tuple
        Elevation and Azimuth in degrees
    """

    pixel_coords = np.array([x_px, y_px])

    # First, we need to undistort the points
    undistorted_point = cv2.undistortPoints(pixel_coords.reshape(1, 1, 2), camera_matrix, distortion_coeffs, P=camera_matrix)

    # Convert to normalized homogeneous coordinates
    norm_point = np.append(undistorted_point, 1)

    # transform to camera coordinates
    img_to_cam = np.linalg.inv(camera_matrix)
    cam_coords = np.dot(img_to_cam, norm_point)

    # Calculate elevation and azimuth based on the camera coordinates
    elevation = np.rad2deg(np.arctan2(-cam_coords[1], cam_coords[2]))
    azimuth = np.rad2deg(np.arctan2(cam_coords[0], cam_coords[2]))

    dva = np.array([azimuth, elevation])

    return dva

def main(root_dir):
    try:
        # load camera intrinsics
        cam_to_img = np.array(json.load(open(root_dir + 'scene_camera.json'))['camera_matrix'])
        # load distortion coefficients
        distortion = np.array(json.load(open(root_dir + 'scene_camera.json'))['distortion_coefficients'])
        # load some example gaze data in px and deg
        gaze_data = pd.read_csv(root_dir + 'gaze.csv')
        # get the columns 'gaze x [px]' and 'gaze y [px]' (only the first 20 rows)
        gaze_px = gaze_data[['gaze x [px]', 'gaze y [px]']].values[:20]
        # get the columns azimuth [deg], elevation [deg] (only the first 20 rows)
        gaze_deg = gaze_data[['azimuth [deg]', 'elevation [deg]']].values[:20]
    except:
        print(f'Error: could not load camera intrinsics or gaze data from {root_dir}\nCheck that the following files exist inside root_dir:\n\t- scene_camera.json\n\t- gaze.csv')
        return

    # apply the projection function to the gaze data
    gaze_px_pred = np.array([dva_to_px(r[1], r[0], cam_to_img, distortion) for r in gaze_deg])
    # apply the inverse projection function to the gaze data
    # px_coords = np.array([np.array(r[0], r[1]).reshape(1,2) for r in gaze_px])
    gaze_deg_pred = np.array([px_to_dva(r[0], r[1], cam_to_img, distortion) for r in gaze_px])

    # compare the gaze px from the .csv to the px calculated from the camera intrinsics
    # seems like pupil-labs rounded to 3 decimal places
    print('gaze px from .csv vs calculated px from camera intrinsics'
          '\n---------------------------------------------------------')
    for i in range(len(gaze_px_pred)):
        print('gaze px: ', gaze_px[i], 'calculated px: ', np.round(gaze_px_pred[i], 3), 'abs error: ', np.abs(gaze_px[i] - np.round(gaze_px_pred[i], 3)))

    print('\n')
    # compare the gaze angles from the .csv to the angles calculated from the camera intrinsics
    # there is a small error, but it is likely due to the rounding of the px values in the .csv
    print('gaze angles from .csv vs calculated angles from camera intrinsics'
            '\n-----------------------------------------------------------------')
    for i in range(len(gaze_deg_pred)):
        print('gaze angles: ', gaze_deg[i], 'calculated angles: ', gaze_deg_pred[i], 'abs error: ', np.abs(gaze_deg[i] - gaze_deg_pred[i]))

if __name__ == '__main__':
    # use argparse to get the root_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='root directory of the data')
    args = parser.parse_args()
    root_dir = args.root_dir
    
    # run the main function to print the results
    main(root_dir)
