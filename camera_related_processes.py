import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader, cpu, gpu
from scipy.ndimage import gaussian_filter
import cv2

def contour_centers(contours, threshold):
    centers = []
    new_contours = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"]>threshold:
            # calculate the centroi coordinates
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
            new_contours.append(contour)
    centers = np.array(centers)

    return centers, new_contours

def filter_contours(contours):
    # get the centers of all contours
    contour_mass_threshold = 0
    centers, contours = contour_centers(contours, contour_mass_threshold)

    # merge contours that are close to each other
    merged_contours = []
    for id in range(len(contours)):
        distances = np.linalg.norm(centers-centers[id], axis=1)
        to_merge = np.where((distances<50) & (distances!=0))[0]
        if any(to_merge):
            merged_contour = contours[id]
            for idx in to_merge:
                merged_contour = np.concatenate([merged_contour, contours[idx]])
            merged_contours.append((merged_contour))
        else:
            merged_contours.append((contours[id]))

    # filter contours based on the mass size 
    contour_mass_threshold = 100
    centers, filtered_contours = contour_centers(merged_contours, contour_mass_threshold)

    return centers, filtered_contours

def get_target_marker(frame):
    # convert colors of the frame
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # color boundary
    lower = np.array([30, 60, 60])
    upper = np.array([60, 255, 255])

    # mask the color 
    mask = cv2.inRange(frame_hsv, lower, upper)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None, None
    
    # filter contours 
    centers, filtered_contours = filter_contours(contours)
    
    # remain only the marker that's closest to the center
    frame_center = np.array([frame.shape[1]/2, frame.shape[0]/2], dtype=int)
    center_idx = np.argmin(np.linalg.norm(centers-frame_center, axis=1))
    target_marker = filtered_contours[center_idx]
    target_center = centers[center_idx]

    return target_center, target_marker

def grab_markers(video, cam_to_img, distortion):
    # iterate frames and detect target markers
    center_markers = pd.DataFrame(columns=["frame", "center x", "center y", "marker"])
    for f in range(len(video)):
        frame = video[f].asnumpy()
        undistorted_frame = cv2.undistort(frame, cam_to_img, distortion)
        target_center, target_marker = get_target_marker(undistorted_frame)

        if target_center is not None:
            center_markers.loc[len(center_markers)] = [int(f), target_center[0], target_center[1], target_marker]
        else:
            center_markers.loc[len(center_markers)] = [int(f), None, None, None]

    return center_markers

def get_median_contour(contour_list, threshold):
    #### the code below is written by Hyong-Kyu Song ####
    contours = np.squeeze(np.concatenate(contour_list))
    _, x_max = np.min(contours[:,0]), np.max(contours[:,0])
    _, y_max = np.min(contours[:,1]), np.max(contours[:,1])
    
    # create an image that marks the contour
    # plt.figure(figsize=(10, 6))   # uncomment this line if you want to see the image of overlapped contours
    aggregate_image = np.zeros((y_max + 1, x_max + 1))
    
    for end_marker in contour_list:
        coordinates = end_marker.squeeze(1)
        
        # Apply aggregation on plot
        x, y = zip(*coordinates)
        # plt.fill(x, y, color='skyblue', alpha=0.02)  # Filling the shape with a color
        
        # Apply aggregation on array itself
        temp_image = np.zeros((y_max + 1, x_max + 1))
        cv2.fillPoly(temp_image, pts=[coordinates], color=(1))
        aggregate_image += temp_image
    
    # Get a binary mask from aggregated array
    mask = aggregate_image >= max(1, int(92 * threshold))
    mask = (mask * 255).astype(np.uint8)
    print(f"Binary mask info: {mask.shape}, {mask.dtype}, {np.unique(mask)}")

    # if threshold is too high, lower the threshold
    while len(np.unique(mask)) == 1:
        threshold = threshold - 0.05
        mask = aggregate_image >= max(1, int(92 * threshold))
        mask = (mask * 255).astype(np.uint8)
        print(f"Binary mask info: {mask.shape}, {mask.dtype}, {np.unique(mask)}")
    
    # Find contour (coordinates) from the binary mask
    median_contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Reshape the median contour
    assert len(median_contour) == 1  # Must be single polygon per video
    median_contour: np.ndarray = median_contour[0]
    median_contour = median_contour.squeeze(1)
    assert median_contour.ndim == 2  # [Coordinate #, 2]
    print(f"median_contours.shape: {median_contour.shape}")
    
    return median_contour

def get_marker_size_px(contour):
    # get marker's width and height by taking median of the dots at extreme
    dot_x, dot_y = zip(*contour)
    dot_x, dot_y = np.array(dot_x), np.array(dot_y)
    nBin = 30
    x_intervals = np.linspace(np.min(dot_x), np.max(dot_x), nBin)
    points = [[dot_x[np.where((dot_x>=x_intervals[0])&(dot_x<x_intervals[1]))]], 
              [dot_x[np.where((dot_x>x_intervals[-2])&(dot_x<=x_intervals[-1]))]]]
    x_lims = [np.median(points[0]), np.median(points[1])]

    y_intervals = np.linspace(np.min(dot_y), np.max(dot_y), nBin)
    points = [[dot_y[np.where((dot_y>=y_intervals[0])&(dot_y<y_intervals[1]))]], 
              [dot_y[np.where((dot_y>y_intervals[-2])&(dot_y<=y_intervals[-1]))]]]
    y_lims = [np.median(points[0]), np.median(points[1])]

    marker_size_pixel = [x_lims[1]-x_lims[0], y_lims[1]-y_lims[0]]

    return marker_size_pixel

def px_to_dva(x_px, y_px, camera_matrix, distortion_coeffs, distorted):
    
    pixel_coords = np.array([x_px, y_px])

    # First, we need to undistort the points
    if not distorted:
        undistorted_point = pixel_coords
    else:
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

