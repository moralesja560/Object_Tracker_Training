import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
				check_imshow, non_max_suppression, apply_classifier, \
				scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
				increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
#For SORT tracking
import skimage
from sort import *
from queue import Queue
from threading import Thread
from datetime import datetime
from dotenv import load_dotenv
from urllib.request import Request, urlopen
import json
from urllib.parse import quote

#python test1sp_c.py --weights "C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\YOLO7\yolov7\runs\train\tiny_yolov7\weights\best.pt" --source "rtsp://root:mubea@10.65.68.2:8554/axis-media/media.amp"
# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org python-dotenv -t

#............................... Environment Variables ............................
load_dotenv()
token_Tel = os.getenv('TOK_EN_BOT')
Jorge_Morales = os.getenv('JORGE_MORALES')
Paintgroup = os.getenv('PAINTLINE')

#............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

area1_pointA = (600,30)
area1_pointB = (600,1150)
area1_pointC = (700,30)
area1_pointD = (700,1150)

#vehicles total springs_count variables
springs = []
hangers  = []
springs_count = 0
hr_springs_count= 0
past_hour = 0
modulo_springs_count = 0

"""" Sends the Hour per Hour Report. """
def send_message(user_id, text,token):
	global json_respuesta
	url = f"https://api.telegram.org/{token}/sendMessage?chat_id={user_id}&text={text}"
	#resp = requests.get(url)
	#hacemos la petición
	try:
		#ruta_state = resource_path("images/tele.txt")
		#file_exists = os.path.exists(ruta_state)
		#if file_exists == False:
		#	return
		#else:
		respuesta  = urlopen(Request(url))
		#	pass
	except Exception as e:
		print(f"Ha ocurrido un error al enviar el mensaje: {e}")
	else:
		#recibimos la información
		cuerpo_respuesta = respuesta.read()
		# Procesamos la respuesta json
		json_respuesta = json.loads(cuerpo_respuesta.decode("utf-8"))
		print("mensaje enviado exitosamente")

"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
	bbox_left = min([xyxy[0].item(), xyxy[2].item()])
	bbox_top = min([xyxy[1].item(), xyxy[3].item()])
	bbox_w = abs(xyxy[0].item() - xyxy[2].item())
	bbox_h = abs(xyxy[1].item() - xyxy[3].item())
	x_c = (bbox_left + bbox_w / 2)
	y_c = (bbox_top + bbox_h / 2)
	w = bbox_w
	h = bbox_h
	return x_c, y_c, w, h

"""Simple function that adds fixed color depending on the class"""
def compute_color_for_labels(label):
	color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
	return tuple(color)

"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
	for i, box in enumerate(bbox):
		x1, y1, x2, y2 = [int(i) for i in box]
		x1 += offset[0]
		x2 += offset[0]
		y1 += offset[1]
		y2 += offset[1]
		
		if categories is not None:
			cat = int(categories[i])  
		else:
			0
		id = int(identities[i]) if identities is not None else 0
		color = compute_color_for_labels(id)
		data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
		label = str(id) + ":"+ names[cat]
		(w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
		cv2.rectangle(img, (x1, y1), (x2, y2), (255,144,30), 1)
		#cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
		cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

		midpoint_x = x1+((x2-x1)/2)
		midpoint_y = y1+((y2-y1)/2)
		center_point = (int(midpoint_x),int(midpoint_y))
		midpoint_color = (0,255,0)
		# Here we create the separate counters, one for categorie 0 and one counter for cat 56 (yolov7-tiny class number for a chair.)
		# Find below the yolov7-tiny trained classes. 56 is for a chair.
		"""
		 class names
			names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
		 """
		# We do not mess with the unique ID system, we just separate the counter's storage. 
		if cat ==0:		
			
			if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
				
				midpoint_color = (0,0,255)
				#print('Kategori : '+str(cat))

				#add vehicle count
				if len(springs) > 0:
					if id not in springs:
						springs.append(id)
				else:
					springs.append(id)
		if cat ==56:
			if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
				
				midpoint_color = (0,0,255)

				#add vehicle count
				if len(hangers) > 0:
					if id not in hangers:
						hangers.append(id)
				else:
					hangers.append(id)
			
		cv2.circle(img,center_point,radius=8,color=midpoint_color,thickness=5)
		
	return img
#..............................................................................
"""Function to consume the results"""
def consumer(queue1):
	print('Consumer: Running')
	now = datetime.now()
	acc_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
	todai = int(now.strftime("%d"))
	# consume work
	while True:
		# get a unit of work
		item = queue1.get()
		# check for stop
		if item is None:
			break
		# report
		#if item > 0:
		print(f'received {item}')
		#The main loop only sends data when the hour has changed, so the thread will wait for the data input to report.
		now = datetime.now()
		hora_consumer = int(now.strftime("%H"))
		acc_hr[int(hora_consumer)] = item
		accumlated_consumer = sum(acc_hr.values())
		#send_message(Paintgroup,quote(f"Hola! Soy una prueba de reporte de hora {hora_consumer-1} - {hora_consumer}: \nHan pasado {item:,} piezas \nAcumulado día: {accumlated_consumer:,} piezas"),token_Tel)
		#I do not expect this thread to fail, so there is no recovery data. 
		if todai != int(now.strftime("%d")):
			acc_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
			todai = int(now.strftime("%d"))
		queue1.task_done()
		print(f"queue processed. {queue1.qsize()}")


			
	# all done
	print('Consumer: Done')

def detect(queue1,save_img=False):
	source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
	save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://', 'https://'))
	
	#Initialize consumer watchdog
	hr_springs_count= 0
	past_hour = 0
	# Check the date and time.
	now = datetime.now()
	hour = int(now.strftime("%H"))
	counter_n = 0
	#springs per minute.
	# we collect time, let's say 0 and the actual springs
	spm_time_1=0
	spm_1 = 0
	# then we wait for the 2000 to pass and select the new time and the new amount of springs.
	spm_time_2 = 0
	spm_2 = 0
	spm = 0

	#.... Initialize SORT .... 
	#......................... 
	sort_max_age = 5 
	sort_min_hits = 2
	sort_iou_thresh = 0.2
	sort_tracker = Sort(max_age=sort_max_age,
					   min_hits=sort_min_hits,
					   iou_threshold=sort_iou_thresh) 
	#......................... 
	# Directories
	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	# Initialize
	set_logging()
	device = select_device(opt.device)
	half = device.type != 'cpu'  # half precision only supported on CUDA
	half = False

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size

	if trace:
		model = TracedModel(model, device, opt.img_size)

	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	if webcam:
		view_img = check_imshow()
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=imgsz, stride=stride)
	else:
		dataset = LoadImages(source, img_size=imgsz, stride=stride)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	# Run inference
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
	old_img_w = old_img_h = imgsz
	old_img_b = 1

	count_vehicle = 0
	count_vehicle2 = 0

	t0 = time.time()
	for path, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Warmup
		if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
			old_img_b = img.shape[0]
			old_img_h = img.shape[2]
			old_img_w = img.shape[3]
			for i in range(3):
				model(img, augment=opt.augment)[0]

		# Inference
		t1 = time_synchronized()
		pred = model(img, augment=opt.augment)[0]
		t2 = time_synchronized()

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t3 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			if webcam:  # batch_size >= 1
				p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			else:
				p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # img.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				#..................USE TRACK FUNCTION....................
				#pass an empty array to sort
				dets_to_sort = np.empty((0,6))
				
				# NOTE: We send in detected object class too
				for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
					dets_to_sort = np.vstack((dets_to_sort, 
								np.array([x1, y1, x2, y2, conf, detclass])))
				
				# Run SORT
				tracked_dets = sort_tracker.update(dets_to_sort)
				tracks =sort_tracker.getTrackers()
				
				#print('Tracked Detections : '+str(len(tracked_dets)))
				
				#loop over tracks
				'''
				for track in tracks:
					# color = compute_color_for_labels(id)
					#draw tracks

					[cv2.line(im0, (int(track.centroidarr[i][0]),
									int(track.centroidarr[i][1])), 
									(int(track.centroidarr[i+1][0]),
									int(track.centroidarr[i+1][1])),
									(0,255,0), thickness=1) 
									for i,_ in  enumerate(track.centroidarr) 
										if i < len(track.centroidarr)-1 ] 
				'''
				
				# draw boxes for visualization
				if len(tracked_dets)>0:
					bbox_xyxy = tracked_dets[:,:4]
					identities = tracked_dets[:, 8]
					categories = tracked_dets[:, 4]
					draw_boxes(im0, bbox_xyxy, identities, categories, names)
					#print('Bbox xy count : '+str(len(bbox_xyxy)))
				#........................................................
				
			# Print time (inference + NMS)
			#print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

			cv2.line(im0,area1_pointA,area1_pointB,(0,255,0),2)
			cv2.line(im0,area1_pointC,area1_pointD,(0,255,0),2)

			color = (255,0,0)
			thickness = 2
			fontScale = 1
			font = cv2.FONT_HERSHEY_SIMPLEX
			org = (160,470)
			org2 = (160,org[1]-50)
			org3 = (160,org[1]+50)
			
			if (count_vehicle == 0):
				springs_count = len(springs)
			else:
				if (springs_count < 100):
					springs_count = len(springs)
				else:
					springs_count = modulo_springs_count + len(springs)
					#hr_springs_count = modulo_springs_count + len(springs)
					if(len(springs)%100 == 0):
						modulo_springs_count = modulo_springs_count + 100
						springs.clear()

			# Other category springs_count
			if (count_vehicle2 == 0):
				hangers_count = len(hangers)
			else:
				if (hangers_count < 100):
					hangers_count = len(hangers)
				else:
					hangers_count = modulo_hangers_count + len(hangers)
					if(len(hangers)%100 == 0):
						modulo_hangers_count = modulo_hangers_count + 100
						hangers.clear()

	#---------------Reporting section----------------------#

			hr_springs_count = springs_count - past_hour
			cv2.putText(im0, f"Today Acc Production: {springs_count:,}", org, font, fontScale, color, thickness, cv2.LINE_AA)
			cv2.putText(im0, f"Hour {hour} Production: {hr_springs_count:,}. Actual SPM {spm:.2f}", org2, font, fontScale, (140,14,140), thickness, cv2.LINE_AA)
			#cv2.putText(im0, f"category 56 {hangers_count}", org3, font, fontScale, (140,14,140), thickness, cv2.LINE_AA)
			counter_n +=1
			# if we check every 15th frame in a 30FPS framerate source, that means we're talking of 2 fps
			# every 2000 iterations, we pass a variable to the reporting thread.
			if counter_n % 1500 ==0:
				#check for actual timestamp
				now = datetime.now()
				times = now.strftime("%d-%m-%y %H:%M:%S")
				if spm_time_1 == 0:
					spm_time_1 = time.time()
					spm_1 = springs_count
					spm = 0
				else:
					spm_time_2 = time.time()
					spm_2 = springs_count
					spm = (spm_2-spm_1)/int((spm_time_2-spm_time_1))*60
					spm_time_1 = spm_time_2
					spm_1 = spm_2

				print(f"sending report {times}: hour is {hour} and actual hour is {int(now.strftime('%H'))}")
				#at the start of this loop, we stored the timestamp. Then we compare it against thte actual timestamp
				if hour != int(now.strftime("%H")):
					if hr_springs_count == 0:
						queue1.put(hr_springs_count+1)
						print(f"Queue sent: {queue1.qsize()}")
					else:
						queue1.put(hr_springs_count)
						print(f"Queue sent: {queue1.qsize()}")
					hour = int(now.strftime("%H"))
					#the springs_count var will not stop
					past_hour = springs_count
					hr_springs_count = 0
				counter_n = 0
			
			
			# Stream results
			if view_img:
				cv2.imshow(str(p), im0)
				cv2.waitKey(1)  # 1 millisecond

			# Save results (image with detections)
			if save_img:
				if dataset.mode == 'image':
					cv2.imwrite(save_path, im0)
					print(f" The image with the result is saved in: {save_path}")
				else:  # 'video' or 'stream'
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer
						if vid_cap:  # video
							fps = vid_cap.get(cv2.CAP_PROP_FPS)
							w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
							h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						else:  # stream
							fps, w, h = 30, im0.shape[1], im0.shape[0]
							save_path += '.mp4'
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
					vid_writer.write(im0)

	if save_txt or save_img:
		s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
		#print(f"Results saved to {save_dir}{s}")

	print(f'Done. ({time.time() - t0:.3f}s)')
	queue1.put(None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
	parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='object_tracking', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
	opt = parser.parse_args()
	print(opt)
	#check_requirements(exclude=('pycocotools', 'thop'))
	queue1 = Queue()
	#start consumer 
	consumer = Thread(target=consumer, args=(queue1,),daemon=True)
	consumer.start()
	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov7.pt']:
				detect(queue1)
				strip_optimizer(opt.weights)
		else:
			detect(queue1)