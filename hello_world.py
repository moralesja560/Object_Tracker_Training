# first, import all necessary modules
from pathlib import Path
import argparse
import blobconverter
import cv2
import depthai
import numpy as np
import json
# python hello_world.py --model "ready_blobs\tiny_y7\best_openvino_2022.1_6shave.blob" -c "ready_blobs\tiny_y7\best.json"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
					default= r'', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
					default=r'', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
	raise ValueError("Path {} does not exist!".format(configPath))

# get model path
nnPath = args.model
if not Path(nnPath).exists():
	raise ValueError("Please select a blob file model")

with configPath.open() as f:
	config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
	W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
# sync outputs
syncNN = True


# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()
# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(W,H)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

# Next, we want a neural network that will produce the detections
detection_nn = pipeline.create(depthai.node.YoloDetectionNetwork)
#Some camera parameters.
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(40)

# Network specific settings
detection_nn.setConfidenceThreshold(confidenceThreshold)
detection_nn.setNumClasses(classes)
detection_nn.setCoordinateSize(coordinates)
detection_nn.setAnchors(anchors)
detection_nn.setAnchorMasks(anchorMasks)
detection_nn.setIouThreshold(iouThreshold)
detection_nn.setBlobPath(nnPath)
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)

# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(detection_nn.input)
# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)


#OpenCV trackbar section
#def empty(a):
#	pass

#cv2.namedWindow("Trackbars")
#cv2.resizeWindow("Trackbars",(640,240))
#cv2.createTrackbar("0 factor","Trackbars",-50,50,empty)
#cv2.createTrackbar("1 factor","Trackbars",-50,50,empty)

spring = 0


# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
	# From this point, the Device will be in "running" mode and will start sending data via XLink

	# To consume the device results, we get two output queues from the device, with stream names we assigned earlier
	q_rgb = device.getOutputQueue("rgb")
	q_nn = device.getOutputQueue("nn")

	# Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
	frame = None
	detections = []

	# Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
	# receive the actual position of the bounding box on the image
	def frameNorm(frame, bbox):
		normVals = np.full(len(bbox), frame.shape[0])
		normVals[::2] = frame.shape[1]
		return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

	# Main host-side application loop
	while True:
		# we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
		in_rgb = q_rgb.tryGet()
		in_nn = q_nn.tryGet()
		#zero_factor = cv2.getTrackbarPos("0 factor","Trackbars")
		#uno_factor = cv2.getTrackbarPos("1 factor","Trackbars")

		if in_rgb is not None:
			# If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
			frame = in_rgb.getCvFrame()

		if in_nn is not None:
			# when data from nn is received, we take the detections array that contains mobilenet-ssd results
			detections = in_nn.detections

		if frame is not None:
			for detection in detections:
				box_coordinate1 = (100,100) #Top left of rectangle
				box_coordinate2 = (350,350) #Bottom right of rectangle
				# for each bounding box, we first normalize it to match the frame size
				bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
				centroid = (int((bbox[0]+((bbox[2]-bbox[0])/2))), int(bbox[1]+((bbox[3]-bbox[1])/2)))
				cv2.circle(frame, centroid, 0, (113,31,154), 5)
				# and then draw a rectangle on the frame to show the actual result
				cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (113,31,154), 1)
				cv2.putText(frame, "spring", (bbox[0] + 10, bbox[1] - 25), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (113,31,154))
				cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10 + 10, bbox[1] - 25 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (113,31,154))
				cv2.rectangle(frame, (box_coordinate1[0],box_coordinate1[1]), (box_coordinate2[0],box_coordinate2[1]), (113,31,154), 2)
				if (centroid[0] > box_coordinate1[0]) and (centroid[0]<box_coordinate2[0]) and (centroid[1] > box_coordinate1[1]) and (centroid[1]<box_coordinate2[1]) :
					spring +=1
				# After all the drawing is finished, we show the frame on the screen
			cv2.putText(frame, f"Springs: {spring}",(2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (113,31,154))
			cv2.imshow("preview", frame)
			spring = 0

		# at any time, you can press "q" and exit the main loop, therefore exiting the program itself
		if cv2.waitKey(1) == ord('q'):
			break