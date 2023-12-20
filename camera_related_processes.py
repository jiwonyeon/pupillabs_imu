import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader, cpu, gpu
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

def find_median_contour(contours):
    xlims = [np.min(contours[:,0]), np.max(contours[:,0])]
    ylims = [np.min(contours[:,1]), np.max(contours[:,1])]
    width = xlims[1]-xlims[0]+1
    height = ylims[1]-ylims[0]+1

    # create a heatmap-like image
    h_image = np.zeros((height, width), dtype=np.uint8)
    for x, y in contours:
        h_image[y-ylims[0], x-xlims[0]] += 1

    # apply gaussian filter 
    sigma = 2
    from scipy.ndimage import gaussian_filter
    h_image_smoothed = gaussian_filter(h_image.astype(float), sigma=sigma)

    # threshold the smoothed image to get a binary mask
    binary_mask = h_image_smoothed > 0.6

    # apply morphological closing to enhance connected regions
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    h_morpho = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # find median contour
    median_contour, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    median_contour = np.squeeze(np.concatenate(median_contour))



