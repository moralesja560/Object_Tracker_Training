import depthai as dai
import cv2
pipeline = dai.Pipeline()
nn = pipeline.create(dai.node.NeuralNetwork)
cam = pipeline.createColorCamera()
cam.setPreviewSize(256,256) 
cam.setInterleaved(False)

nn.setBlobPath(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\Object_Tracker_Training\blobs\pizza (1).blob')
nn.getOutputs()
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(nn.input)

# Send NN out to the host via XLink
nnXout = pipeline.create(dai.node.XLinkOut)
nnXout.setStreamName("nn")
nn.out.link(nnXout.input)

with dai.Device(pipeline) as device:
	qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

#You can now decode the output of your NN
# Pipeline is defined, now we can connect to the device
	while True:
		nnData = qNn.get() # Blocking
		# NN can output from multiple layers. Print all layer names:
		# Get layer named "Layer1_FP16" as FP16
		layer1Data = nnData.getLayerFp16("model/dense/BiasAdd")
		print(layer1Data)
		if cv2.waitKey(1) == ord('q'):
			break