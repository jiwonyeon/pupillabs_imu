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
    threshold = 0 
    centers, contours = contour_centers(contours, threshold)

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
    threshold = 100
    centers, filtered_contours = contour_centers(merged_contours, threshold)

    return filtered_contours



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
    if not any(contours):
        return None
    
    # filter contours 
    filtered_contours = filter_contours(contours)
    
    cv2.drawContours(frame_masked, filtered_contours, -1, (0, 255, 0), 3)

    

def grab_markers(video):
    # iterate frames and detect target markers
    for f in range(len(video)):
        frame = video[f].asnumpy()
        markers = get_target_marker(frame)

        if markers is not None:
            pass

    # return markers
