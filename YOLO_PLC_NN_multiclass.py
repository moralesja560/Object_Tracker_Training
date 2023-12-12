#------------------Author Info----------------------#
#		   The AI Paintline Spring Counter
# Designed and developed by: Ing Jorge Alberto Morales, MBA
# Automation Project Sr Engineer for Mubea Coil Springs Mexico
#			Jorge.Morales@mubea.com / 8445062027
#---------------------------------------------------#




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
import pyads
import sys
import tensorflow as tf

#python test1sp_c.py --weights "C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\YOLO7\yolov7\runs\train\tiny_yolov7\weights\best.pt" --source "rtsp://root:mubea@10.65.68.2:8554/axis-media/media.amp"
# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org python-dotenv -t

"""
Variable naming rules and programming conventions

1.-CNS, PLC, NN and YLO are the preffixes. When a variable is named NN_springs_count, it means that it only lives in the NN function
	1.1.- The only vars that won't be subject to this convention are the original counters from YOLO.
2.- springs and hangers are the only allowed categories
3.- hr and dai will be the suffixes that will signal they're hourly or daily vars.
4.- ALL the function's internal vars are subject to be renamed to prevent overlapping.
"""


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

area1_pointA1 = (300,1)
area1_pointB1 = (300,100)
area1_pointC1 = (400,1)
area1_pointD1 = (400,100)

#vehicles total springs_count variables
springs = []
hangers  = []


"""References when pyinstaller"""
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

""" References to MyDocuments"""
def My_Documents(location):
	import ctypes.wintypes
		#####-----This section discovers My Documents default path --------
		#### loop the "location" variable to find many paths, including AppData and ProgramFiles
	CSIDL_PERSONAL = location       # My Documents
	SHGFP_TYPE_CURRENT = 0   # Get current, not default value
	buf= ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
	ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
	#####-------- please use buf.value to store the data in a variable ------- #####
	#add the text filename at the end of the path
	temp_docs = buf.value
	return temp_docs


"""" Sends the Hour per Hour Report. """
def send_message(user_id, text,token):
	if opt.noTele:
		print("CNS: Disable Consumer Messaging Service")
		return
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
	# Initialize variable to store the hanger photo
	img_copy = img.copy()
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
		cv2.rectangle(img, (x1, y1), (x2, y2), (255,144,30), 2)
		#cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
		cv2.putText(img, label, (x1, y1 + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)

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
		if cat ==1:
			if (midpoint_x > area1_pointA1[0] and midpoint_x < area1_pointD1[0]) and (midpoint_y > area1_pointA1[1] and midpoint_y < area1_pointD1[1]):
				if not opt.noNN:
					#store image
					file_exists = os.path.exists(f"NN_results\photo_{cat}_{id}.jpg")
					if not file_exists:
						address = str(f"NN_results\photo_{cat}_{id}.jpg")
						cv2.imwrite(f"NN_results\photo_{cat}_{id}.jpg", img_copy)
						queue3.put(address)

				midpoint_color = (0,0,255)
				#add hanger count
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
	print('++++++++++++++++++++++++++++++++++ Consumer: Running')
	now = datetime.now()
	acc_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
	hang_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
	NN_todai = int(now.strftime("%d"))
	# consume work
	while True:
		# get a unit of work
		springs_hr,hangers_hr,CNS_empty_h,CNS_empty_d = queue1.get()
		# check for stop
		if springs_hr is None:
			break
		print(f'received {springs_hr}/{hangers_hr}')
		#The main loop only sends data when the hour has changed, so the thread will wait for the data input to report.
		now = datetime.now()
		hora_consumer = int(now.strftime("%H"))
		
		#---------------Storage Section---------------------#
		if hora_consumer == 0:
			initial_hour = 23
		else:
			initial_hour = hora_consumer-1
		#we store the springs in the appropiate hour slot.
		acc_hr[int(initial_hour)] = int(springs_hr)
		# we store the hangers in the appropiate hour slot.
		hang_hr[int(initial_hour)] = int(hangers_hr)
		#---------------Telegram Report Section---------------------#
		# we sum the values
		acc_springs = sum(acc_hr.values())
		acc_hangers = sum(hang_hr.values())
		#---------------Telegram Report Section---------------------#
	
		send_message(Paintgroup,quote(f"Reporte de Hora {initial_hour} - {hora_consumer}: \nPiezas: {springs_hr:,} \nGancheras detectadas: {hangers_hr:,} \nHuecos: {CNS_empty_h:,} \nEff Pintura: {(hangers_hr/79):.2%} \nEff Prod: {springs_hr/(hangers_hr*20):.2%}"),token_Tel)
		send_message(Paintgroup,quote(f"Acumulado Hoy: \nTotal Piezas: {acc_springs:,} \nTotal Gancheras: {acc_hangers:,} \nTotal Huecos: {CNS_empty_d:,}"),token_Tel)

		#I do not expect this thread to fail, so there is no recovery data. 
		queue1.task_done()
		print(f"CONS processed. {queue1.qsize()}")
		if NN_todai != int(now.strftime("%d")):
			acc_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
			hang_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
			NN_todai = int(now.strftime("%d"))

			
	# all done
	print('Consumer: Done')

def PLC_comms(queue2,plc):
	print('++++++++++++++++++++++++++++++++++ PLC: Running')
	try:
		plc.open()
		plc.set_timeout(2000)
		YOLO_counter = plc.get_handle('.YOLO_counter_UINT')
		var_handle_full_hr = plc.get_handle('SCADA.Full_hooks_hr')
		var_handle_full_day = plc.get_handle('SCADA.Full_hooks_day')
		var_handle_empty_hr = plc.get_handle('SCADA.Empty_hooks_hr')
		var_handle_empty_day = plc.get_handle('SCADA.Empty_hooks_day')
		var_handle_actual_hook = plc.get_handle('SCADA.This_hook')
	except Exception as e:
			print(f"Starting error: {e}")
			time.sleep(5)
			plc, YOLO_counter, var_handle_full_hr, var_handle_full_day, var_handle_empty_hr, var_handle_empty_day,var_handle_actual_hook= aux_PLC_comms()
	while True:
		# get a unit of work
		PLC_springs,PLC_full_hr,PLC_empty_hr,PLC_full_d,PLC_empty_d,PLC_this_hook = queue2.get()

		# check for stop
		if PLC_springs is None:
			#PLC release and break
			plc.release_handle(YOLO_counter)
			plc.release_handle(var_handle_full_hr)
			plc.release_handle(var_handle_full_day)
			plc.release_handle(var_handle_empty_hr)
			plc.release_handle(var_handle_empty_day)
			plc.close()
			break
		#it's time to work.
		try:
			plc.write_by_name("", int(PLC_springs), plc_datatype=pyads.PLCTYPE_UINT,handle=YOLO_counter)
			#llenas por h
			plc.write_by_name("", int(PLC_full_hr), plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_full_hr)
			#llenas por d
			plc.write_by_name("", int(PLC_full_d), plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_full_day)
			#vacias por h
			plc.write_by_name("", int(PLC_empty_hr), plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_empty_hr)
			#vacias por d
			plc.write_by_name("", int(PLC_empty_d), plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_empty_day)
			#this hook
			print(f"                                     This_hook reported with {PLC_this_hook}")
			if PLC_this_hook > 0:
				plc.write_by_name("", int(PLC_this_hook), plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_actual_hook)

		except Exception as e:
			print(f"Could not update in PLC: error {e}")
			queue2.task_done()
			plc, YOLO_counter, var_handle_full_hr, var_handle_full_day, var_handle_empty_hr, var_handle_empty_day,var_handle_actual_hook= aux_PLC_comms()
			continue
		else:
			print(f"PLC processed. {queue2.qsize()} remaining in queue")
			queue2.task_done()
			
def aux_PLC_comms():

	while True:
		try:
			plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
			plc.open()
			YOLO_counter = plc.get_handle('.YOLO_counter_UINT')
			var_handle_full_hr = plc.get_handle('SCADA.Full_hooks_hr')
			var_handle_full_day = plc.get_handle('SCADA.Full_hooks_day')
			var_handle_empty_hr = plc.get_handle('SCADA.Empty_hooks_hr')
			var_handle_empty_day = plc.get_handle('SCADA.Empty_hooks_day')
			var_handle_actual_hook = plc.get_handle('SCADA.This_hook')
		except:
			print(f"Auxiliary PLC: Couldn't open")
			time.sleep(4)
			continue
		else:
			plc.open()
			print("Success PLC")
			return plc, YOLO_counter, var_handle_full_hr, var_handle_full_day, var_handle_empty_hr, var_handle_empty_day, var_handle_actual_hook

def NN_process(q,NNmodel,q4):
	# Q is to receive the string where ill get the phot
	# Q4 is to send the info to YOLO
	#previous_address = ""
	NN_full_hangers_hr = 0
	NN_empty_hangers_hr = 0
	NN_full_hangers_dai = 0
	NN_empty_hangers_dai = 0
	NN_actual_hook = 0
	now = datetime.now()
	while True:
		# We receive an unit of work
		img_address = q.get()
		
		if img_address == None:
			break
		# IF we receive a code, then we reset the variable.
		if img_address == "N400":
			NN_full_hangers_hr =0
			NN_empty_hangers_hr =0
			print("NN: Reset H Vars")
			q.task_done()
			continue
		if img_address == "N600":
			NN_full_hangers_dai =0
			NN_empty_hangers_dai =0
			print("NN: Reset D Vars")
			q.task_done()
			continue
		img = cv2.imread(img_address)
		image = cv2.resize(img,dsize=(224,224), interpolation = cv2.INTER_CUBIC) 
		final_data = NNmodel.predict(np.expand_dims(image, axis=0),verbose=0)
		final_data = final_data.item()
		now = datetime.now()
		times = now.strftime("%d%m%y-%H%M%S")
		#GUARDAMOS LA IMAGEN
		# SI ES 1, ENTONCES LA gch ESTA LLENA
		if final_data >0:
			#cv2.imwrite(resource_path(f'NN_results/full/full-{times}.jpg'), img)
			NN_storage(img,1)
			print(f"NN full image stored with {int(final_data)}")
			NN_full_hangers_hr +=1
			NN_full_hangers_dai +=1
			NN_actual_hook = 1
		else:
			#cv2.imwrite(resource_path(f'NN_results/empty/empty-{times}.jpg'), img)
			NN_storage(img,2)
			print(f"NN empty image stored with {int(final_data)}")
			NN_empty_hangers_hr +=1
			NN_empty_hangers_dai +=1
			NN_actual_hook = 2
		#we delete the original image
		q4.put((NN_full_hangers_hr,NN_empty_hangers_hr,NN_full_hangers_dai,NN_empty_hangers_dai,NN_actual_hook))
		q.task_done()
		print(f"NN processed: Input Q3: {q.qsize()} Output Q4: {q4.qsize()}")
		time.sleep(10)
		os.remove(img_address)
		
def NN_storage(img,command):
	"""
	This function is to store images in the correct folder to ensure trazability
	The folder is in Mydocuments
	Documents/Paintline_Evidence/<day>/<hour>/full
	Documents/Paintline_Evidence/<day>/<hour>/empty
	"""
	# a new image has been received, check if folder exists
	# Step 1: path first step
	mis_docs = My_Documents(5)
	now = datetime.now()
	#date
	NN_folder_name= "\\Paintline_Evidence\\" + now.strftime("%d-%m-%y--%H")
	pd_ruta = str(mis_docs) + NN_folder_name
	pd_ruta_full = pd_ruta + r'\\full'
	pd_ruta_empty = pd_ruta + r'\\empty'
	
	if not os.path.isdir(pd_ruta):
		os.makedirs(pd_ruta)
		os.makedirs(pd_ruta_full)
		os.makedirs(pd_ruta_empty)
	
	times = now.strftime("%d%m%y-%H%M%S")
	if command == 1:
		#full hanger
		cv2.imwrite(pd_ruta_full+f"\{times}.jpg",img)
	elif command == 2:
		cv2.imwrite(pd_ruta_empty+f"\{times}.jpg",img)
	return

def detect(queue1,save_img=False):
	source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
	save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://', 'https://'))
	
	#Initialize consumer watchdog
	hr_springs_count= 0
	past_springs_hour = 0
	past_hanger_hr = 0
	past_springs_dai = 0
	past_hanger_dai = 0
	modulo_hangers_count = 0
	modulo_springs_count = 0
	# Check the date and time.
	now = datetime.now()
	hour = int(now.strftime("%H"))
	YLO_todai = int(now.strftime("%d"))
	counter_n = 0
	#springs per minute.
	# we collect time, let's say 0 and the actual springs
	spm_time_1=0
	spm_1 = 0
	# then we wait for the 2000 to pass and select the new time and the new amount of springs.
	spm_time_2 = 0
	spm_2 = 0
	spm = 0
	# Vars that come from the PLC.
	# Vars that come from the NN
	YLO_full_hangers_hr = 0
	YLO_empty_hangers_hr = 0
	YLO_full_hangers_dai = 0
	YLO_empty_hangers_dai = 0
	YLO_this_hook = 0


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
			# Line drawing
			cv2.line(im0,area1_pointA,area1_pointB,(0,255,0),2)
			cv2.line(im0,area1_pointC,area1_pointD,(0,255,0),2)
			cv2.line(im0,area1_pointA1,area1_pointB1,(233,242,58),2)
			cv2.line(im0,area1_pointC1,area1_pointD1,(233,242,58),2)

			color = (255,0,0)
			thickness = 1
			fontScale = 0.6
			font = cv2.FONT_HERSHEY_SIMPLEX
			#For left side
			left_first_row = (200,300)
			left_second_row = (left_first_row[0],left_first_row[1]+35)
			left_third_row = (left_first_row[0],left_first_row[1]+70)
			left_fourth_row = (left_first_row[0],left_first_row[1]+105)
			left_fifth_row = (left_first_row[0],left_first_row[1]+140)
			#For right side
			right_first_row = (left_first_row[0]+300,300)
			right_second_row = (left_first_row[0]+300,left_first_row[1]+35)
			right_third_row = (left_first_row[0]+300,left_first_row[1]+70)

			
			if (count_vehicle == 0):
				springs_count = len(springs)
			else:
				if (springs_count < 100):
					springs_count = len(springs)
				else:
					springs_count = modulo_springs_count + len(springs)
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
			#hourly reset
			hr_springs_count = springs_count - past_springs_hour
			hr_hangers_count = hangers_count - past_hanger_hr
			#daily reset
			dai_springs_count = springs_count - past_springs_dai
			dai_hangers_count = hangers_count - past_hanger_dai
			#---------------Left Side: Hangers
			# Total Hangers
			cv2.putText(im0, f"Hangers Today: {dai_hangers_count}", left_first_row, font, fontScale, (140,14,140), thickness, cv2.LINE_AA)
			# This Hour
			cv2.putText(im0, f"Hangers in Hour {hour}: {hr_hangers_count}", left_second_row, font, fontScale, (140,14,140), thickness, cv2.LINE_AA)
			#Full/Empty Hangers
			cv2.putText(im0, f"Hangers Full/Empty Hour {hour}: {YLO_full_hangers_hr}/{YLO_empty_hangers_hr}", left_third_row, font, fontScale, (140,14,140), thickness, cv2.LINE_AA)
			#Full/Empty Hangers
			cv2.putText(im0, f"Hangers Full/Empty Today: {YLO_full_hangers_dai}/{YLO_empty_hangers_dai}", left_fourth_row, font, fontScale, (140,14,140), thickness, cv2.LINE_AA)			
			#------------Right Side: Springs
			# Total Springs
			cv2.putText(im0, f"Springs Today: {dai_springs_count:,}", right_first_row, font, fontScale, (29,99,4), thickness, cv2.LINE_AA)
			# This Hour + Speed
			cv2.putText(im0, f"Springs in Hour {hour}: {hr_springs_count:,}. Actual SPM {spm:.2f}", right_second_row, font, fontScale, (29,99,4), thickness, cv2.LINE_AA)
			
			
			counter_n +=1
			# if we check every 15th frame in a 30FPS framerate source, that means we're talking of 2 fps
			# every 2000 iterations, we pass a variable to the reporting thread.
			if counter_n % 150 ==0:
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
				print(f"YOLO Update {times}")

				if not opt.noNN:
					try:
						YLO_full_hangers_hr,YLO_empty_hangers_hr,YLO_full_hangers_dai,YLO_empty_hangers_dai,YLO_this_hook = queue4.get(block=False)
						print(f"NN > YOLO: {queue4.qsize()} remaining. Received this_hook with {YLO_this_hook}")
						queue4.task_done()
					except:
						print("NN did not report this cycle")

				if not opt.noPLC:
					queue2.put((hr_springs_count,YLO_full_hangers_hr,YLO_empty_hangers_hr,YLO_full_hangers_dai,YLO_empty_hangers_dai,YLO_this_hook))	
					print(f"YOLO > PLC: {queue2.qsize()} remaining in queue. SENT this hook {YLO_this_hook}")
					YLO_this_hook =0
			
				#at the start of this loop, we stored the timestamp. Then we compare it against thte actual timestamp
				if hour != int(now.strftime("%H")):
					#In queue1 we report the info to Telegram
					queue1.put((hr_springs_count,hr_hangers_count,YLO_empty_hangers_hr,YLO_empty_hangers_dai))
					print(f"YOLO: Consumer Queue sent: {queue1.qsize()}")					
					hour = int(now.strftime("%H"))
					
					#Then we reset the vars to start the new hour
					past_springs_hour = springs_count
					past_hanger_hr = hangers_count
					hr_springs_count = 0
					hr_hangers_count = 0
					if not opt.noNN:
						queue3.put("N400")
				if YLO_todai != int(now.strftime("%d")):
					if not opt.noNN:
						queue3.put("N600")
					past_springs_dai = springs_count
					past_hanger_dai = hangers_count
					dai_springs_count = 0
					dai_hangers_count = 0
					YLO_todai = int(now.strftime("%d"))		
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

	#Finish Processing and closing threads by sending NONE to the queues.
	print(f'Done. ({time.time() - t0:.3f}s)')
	queue1.put((None,0,0,0))
	if not opt.noPLC:
		queue2.put((None,0,0,0,0,0))
	if not opt.noNN:
		queue3.put(None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.70, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
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
	parser.add_argument('--noPLC', action='store_true', help='Use the counter with no PLC')
	parser.add_argument('--noNN', action='store_true', help='Use the counter with no NN Output')
	parser.add_argument('--noTele', action='store_true', help='Disable Telegram messaging')
	opt = parser.parse_args()
	#print(opt)

	#start consumer and PLC queues
	#Q1 for Telegram Reporting. YOLO -> Consumer Thread
	queue1 = Queue()
	#Q2 is for PLC reporting  YOLO -> PLC 
	queue2 = Queue()
	#Q3 is for YOLO -> NN
	queue3 = Queue()
	#Q4 is for NN -> YOLO
	queue4 = Queue()
	#Verify PLC connection to report.
	if not opt.noPLC:
		try:
			pyads.open_port()
			ams_net_id = pyads.get_local_address().netid
			print(ams_net_id)
			pyads.close_port()
			plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
			plc.set_timeout(2000)
			PLC_thread = Thread(name="hilo_PLC",target=PLC_comms, args=(queue2,plc),daemon=True)
		except:
			print("PLC couldn't be open. Try establishing it first using System Manager")
			sys.exit()
		else:
			PLC_thread.start()
	
	if not opt.noNN:
		print("Loading NN Model....")
		new_model = tf.keras.models.load_model(resource_path(r"model_7_paintline"))
		print("Loaded NN Model")
		NN_thread = Thread(name="hilo_NN",target=NN_process, args=(queue3,new_model,queue4),daemon=True)
		NN_thread.start()

	
	#Start consumer thread
	consumer = Thread(target=consumer, args=(queue1,),daemon=True)
	consumer.start()

	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov7.pt']:
				detect(queue1)
				strip_optimizer(opt.weights)
		else:
			detect(queue1)