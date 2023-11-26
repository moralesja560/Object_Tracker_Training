#!/usr/bin/env python3

from depthai_sdk import OakCamera, TrackerPacket, Visualizer, TextPosition
import depthai as dai
from people_tracker import PeopleTracker
import cv2

pt = PeopleTracker()

with OakCamera() as oak:
	color_cam = oak.create_camera('color')
	#tracker = oak.create_nn(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\Object_Tracker_Training\ready_blobs\tiny_y7\best_openvino_2022.1_6shave.blob', color_cam, tracker=True,nn_type=)
	tracker = oak.create_nn(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\Object_Tracker_Training\ready_blobs\tiny_y7\best_openvino_2022.1_6shave.blob', input=color_cam,nn_type='yolo',tracker=True)
	tracker.config_nn(conf_threshold=0.6)
	tracker.config_tracker(tracker_type=dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM, track_labels=[1])

	def cb(packet: TrackerPacket, vis: Visualizer):
		left, right, up, down = pt.calculate_tracklet_movement(packet.daiTracklets)

		vis.add_text(f"Up: {up}, Down: {down}", position=TextPosition.TOP_LEFT, size=1)
		vis.draw(packet.frame)

		cv2.imshow('People Tracker', packet.frame)



while True:
	oak.visualize(tracker.out.tracker, callback=cb)
	oak.start(blocking=True)
	if cv2.waitKey(1) == ord('q'):
		break