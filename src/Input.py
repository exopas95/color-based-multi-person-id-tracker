# -*- coding: utf-8 -*-
import Constants

# CIE2000 for color difference
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor

# Load K-means clustering for color-mean value
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load DeepGaze
from deepgaze.color_detection import BackProjectionColorDetector

# Load DeepSort
from utils import poses2boxes
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from deep_sort.linear_assignment import min_cost_matching
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.tracker import Tracker as DeepTracker
from deep_sort.detection import Detection
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.iou_matching import iou_cost

# Load OpenPose:
from openpose import pyopenpose as op

import cv2
import sys
import time
import numpy as np
import codecs
import json
from datetime import datetime
import pygame

sys.path.append('/usr/local/python')


class Input():
    def __init__(self, debug=False):
        #from openpose import *
        params = dict()
        params["model_folder"] = Constants.openpose_modelfolder
        params["net_resolution"] = "-1x320"
        self.openpose = op.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()

        max_cosine_distance = Constants.max_cosine_distance
        nn_budget = Constants.nn_budget
        self.nms_max_overlap = Constants.nms_max_overlap
        max_age = Constants.max_age
        n_init = Constants.n_init

        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepTracker(metric, max_age=max_age, n_init=n_init)

        self.capture = cv2.VideoCapture(
            "/home/treenulbo/Develop/live-dance-tracker/youtube_example/BTS_GO.mp4")

        if self.capture.isOpened():         # Checks the stream
            self.frameSize = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        Constants.SCREEN_HEIGHT = self.frameSize[0]
        #Constants.SCREEN_WIDTH = self.frameSize[1]
        Constants.SCREEN_WIDTH = 450

        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('dance_tracker_person_1.avi', fourcc,
                                   30.0, (Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT))

    def getCurrentFrameAsImage(self, p_x1, p_x2):
        frame = self.currentFrame[0: self.currentFrame.shape[0], p_x1: p_x2]
        #frame = self.currentFrame[0 : self.currentFrame.shape[0], 0 : 1920]
        self.out.write(frame)  # write video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pgImg = pygame.image.frombuffer(
            frame.tostring(), frame.shape[1::-1], "RGB")
        return pgImg

    def centroid_histogram(self, clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    def plot_colors(self, hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    def image_color_cluster(self, color_image, person_id, k=5):
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        clt = KMeans(n_clusters=k)
        clt.fit(image)

        # for center in clt.cluster_centers_:
        #     print("p_Id: {} - {}".format(person_id, center))
        # average = color_image.mean(axis=0).mean(axis=0)

        hist = self.centroid_histogram(clt)
        bar = self.plot_colors(hist, clt.cluster_centers_)
        cmax = np.max(hist)
        cmax_index = np.where(cmax == hist)
        d_color = np.round(clt.cluster_centers_[cmax_index], decimals=0)
        d_color = [d_color[0][0], d_color[0][1], d_color[0][2]]

        return d_color

    def color_distance(self, color_1, color_2):
        original_srgb = sRGBColor(
            color_2[0], color_2[1], color_2[2], is_upscaled=True)
        new_srgb = sRGBColor(color_1[0], color_1[1],
                             color_1[2], is_upscaled=True)

        # calculate delta
        original_color = convert_color(original_srgb, LabColor)
        new_color = convert_color(new_srgb, LabColor)
        delta_e = delta_e_cie2000(original_color, new_color)

        return delta_e

    def min_color_distance(self, new_color, original_list):
        new_srgb = sRGBColor(
            new_color[0], new_color[1], new_color[2], is_upscaled=True)
        new_color = convert_color(new_srgb, LabColor)
        min_key = 0
        min_delta = 100000

        for index in original_list:
            original_srgb = sRGBColor(
                original_list[index][0], original_list[index][1], original_list[index][2], is_upscaled=True)
            original_color = convert_color(original_srgb, LabColor)
            delta = delta_e_cie2000(original_color, new_color)
            if delta < min_delta:
                min_delta = delta
                min_key = index

        return min_key

    def cord_distance(self, original_cords, new_cords):
        formula_1 = pow((original_cords[0] - new_cords[0]), 2) + \
            pow((original_cords[1] - new_cords[1]), 2)
        formula_2 = pow((original_cords[2] - new_cords[2]), 2) + \
            pow((original_cords[3] - new_cords[3]), 2)
        formula = (formula_1 + formula_2) / 2
        if formula != 0:
            euclidean_d = math.sqrt(formula)
        else:
            euclidean_d = 0

        # distance too far
        return euclidean_d

    def run(self, p_id, col_list, g_num, start, end):
        result, self.currentFrame = self.capture.read()

        # frame width
        width_start = 0
        width_end = 0

        datum = op.Datum()
        datum.cvInputData = self.currentFrame
        self.openpose.emplaceAndPop([datum])

        #keypoints, self.currentFrame = np.array(datum.poseKeypoints), datum.cvOutputData
        keypoints = np.array(datum.poseKeypoints)

        # Doesn't use keypoint confidence
        # [RShoulder, RElbow, LShoulder, LElbow, MidHip, RHip,Rknee, RAnkle, LHip, LKnee, LAnkle]
        poses = keypoints[:, [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14], :2]

        # Remove zero elemets
        zero_list = []
        for index in range(0, len(poses)):
            if np.sum(poses[index]) == 0:
                zero_list.append(index)
        if len(zero_list) != 0:
            poses = np.delete(poses, zero_list, axis=0)
        # Get containing box for each seen body
        boxes = poses2boxes(poses)
        boxesxywh = [[x1, y1, x2-x1, y2-y1] for [x1, y1, x2, y2] in boxes]
        features = self.encoder(self.currentFrame, boxesxywh)

        def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
        detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in zip(
            boxesxywh, features, poses) if nonempty(bbox)]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes_det, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(self.currentFrame, detections)
        track_list = self.tracker.tracks

        for track in track_list:
            color = None
            if not track.is_confirmed():
                color = [0, 0, 255]
            else:
                color = [255, 255, 255]
            bbox = track.to_tlbr()

            # DeepGaze color detection
            person_key = int(track.track_id)    # Personal Id
            obj_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]),
                       int(bbox[3])]  # object tracker cordinates
            col_box = [int(bbox[0])-2, int(bbox[1])+2, int(bbox[2])+2, int(bbox[1])+2 - int(
                (int(bbox[1]) + 2 - int(bbox[3])) * 0.6)]  # color tracker cordinates

            clear_condition = False  # 박스가 일정 크기 이상
            color_detected = True   # 색깔 발견
            if (col_box[2] - col_box[0]) < 200 and (col_box[2] - col_box[0]) > 80 and (col_box[3] - col_box[1]) > 130:
                clear_condition = True

            try:
                template = self.currentFrame[col_box[1]                                             :col_box[3], col_box[0]:col_box[2]]
                # Defining the deepgaze color detector object
                my_back_detector = BackProjectionColorDetector()
                my_back_detector.setTemplate(template)  # Set the template
                #image_filtered = my_back_detector.returnFiltered(self.currentFrame, morph_opening=True, blur=True, kernel_size=7, iterations=2)

                # 색깔 고정해주기
                if clear_condition == True:
                    dominant_color = self.image_color_cluster(
                        template, person_key)
                    if len(col_list) == 0:
                        col_list[person_key] = dominant_color
                    elif person_key in list(col_list.keys()):
                        # 안가려져있을때
                        if self.color_distance(dominant_color, col_list[person_key]) < 5:
                            col_list[person_key] = dominant_color
                        else:
                            person_key = self.min_color_distance(
                                dominant_color, col_list)
                            dominant_color = col_list[person_key]
                    else:
                        if person_key <= g_num:
                            col_list[person_key] = dominant_color
                else:
                    color_detected = False

                # images_stack = np.hstack((self.currentFrame,image_filtered)) #The images are stack in order
                # cv2.imwrite("personal_color{}.jpg".format(count), images_stack) #Save the image if you prefer
            except:
                color_detected = False

            # 안가려져있을 때
            if person_key > g_num:
                if color_detected == False:
                    dominant_color = [255, 255, 255]
                person_key = self.min_color_distance(dominant_color, col_list)

            if (col_box[2] - col_box[0]) > 40 and (col_box[2] - col_box[0]) < 220:
                # Detectron Object Detector
                cv2.rectangle(
                    self.currentFrame, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), color, 2)
                # cv2.rectangle(self.currentFrame, (500, 100), (600, 230), [0, 0, 0], 2)

                # DeepGaze color detector
                key_list = list(col_list.keys())
                if person_key in key_list and color_detected == True:
                    cv2.rectangle(self.currentFrame, (col_box[0], col_box[1]), (col_box[2], col_box[3]), (
                        col_list[person_key][2], col_list[person_key][1], col_list[person_key][0]), 2)  # Drawing a green rectangle around the template

                # Allocate Id
                cv2.putText(self.currentFrame, "id%s" % (
                    person_key), (obj_box[0], obj_box[1]-20), 0, 5e-3 * 200, (0, 255, 0), 2)

            # update information
            if person_key == p_id and clear_condition == True:
                width_start = int((obj_box[0] + obj_box[2]) / 2) - 225
                width_end = int((obj_box[0] + obj_box[2]) / 2) + 225

        if width_start <= 0 or width_end <= 0 or width_start > 1920 or width_end > 1920:
            width_start = start
            width_end = end

        # Save personal image by frame
        # if result == True and p_id != None:
        #     height = self.currentFrame.shape[0]
        #     width = self.currentFrame.shape[1]
        #     personal_image = self.currentFrame[0 : height, p.PERSON_X1:p.PERSON_X2]
        #     cv2.imwrite('personal_image/person_{}_{}.jpg'.format(p_id, frame_index),personal_image)
        #     frame_index += 1

        # Write result to JSON file
        #json.dump(personal_info, codecs.open("../result/result{}.json".format(current_time), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=2)
        #json.dump(personal_info, codecs.open("../../dance-result/temp/Gashina3_{}.json".format(current_time), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=2)

        # print("Cordinate: ({}, {})".format(width_start, width_end))
        current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        cv2.waitKey(1)

        return width_start, width_end, col_list
