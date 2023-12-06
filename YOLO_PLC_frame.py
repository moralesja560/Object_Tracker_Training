import pyads
import time
import sys
from queue import Queue
from threading import Thread
from utils.datasets import LoadStreams, LoadImages
import cv2
from pathlib import Path

class CustomThread(Thread):
	def capture():
		dataset = LoadStreams(source, img_size=imgsz, stride=stride)
		h=0
		try:
			plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
			plc.set_timeout(2000)
			plc.open()
			hanger_counter = plc.get_handle('.I_Hang_Counter')
		except Exception as e:
			print(f"Starting error: {e}")
			time.sleep(5)
			plc, hanger_counter = aux_PLC_comms()
		for path, img, im0s, vid_cap in dataset:
			i=0
			p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			p = Path(p)
			try:
				cell1 = plc.read_by_name("", plc_datatype=pyads.PLCTYPE_BOOL,handle=hanger_counter)
			except Exception as e:
				print(f"failed to read from PLC: Error {e}")
				continue
			if cell1:
				cv2.imshow(f"{p}", im0)
				cv2.waitKey(100)  # 1 millisecond
				cv2.destroyAllWindows()

			

def aux_PLC_comms():
	while True:
		try:
			plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
			plc.open()
			hanger_counter = plc.get_handle('.I_Hang_Counter')
		except:
			print(f"Auxiliary PLC: Couldn't open")
		else:
			print("Success")
			return plc, hanger_counter


if __name__ == '__main__':

	# connect to the PLC
	try:
		pyads.open_port()
		ams_net_id = pyads.get_local_address().netid
		print(ams_net_id)
		pyads.close_port()
		plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
		plc.set_timeout(1000)
	except:
		print("No se pudo abrir la conexión")
		sys.exit()
	 #open the connection
	#source = "rtsp://root:mubea@10.65.68.2:8554/axis-media/media.amp"
	source = "rtsp://user:user@10.65.68.47/axis-media/media.amp"
	imgsz = 640
	stride = 32
	thread = CustomThread(target=CustomThread.capture(),args=plc)