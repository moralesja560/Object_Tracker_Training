from utils.datasets import LoadStreams, LoadImages
import cv2
import torch
from utils.torch_utils import select_device
from pathlib import Path
import time
from threading import Thread
import queue

class CustomThread(Thread):
	def capture():

		dataset = LoadStreams(source, img_size=imgsz, stride=stride)
		h=0
		for path, img, im0s, vid_cap in dataset:
			i=0
			p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			p = Path(p)
			cv2.imshow(f"{p}", im0)
			cv2.waitKey(1)  # 1 millisecond
			h+=1
			if (h % 20==0):
				#queue1.put(h)
				#print(f"put in queue{h}, frames {frame}")
				pass
				
				
class CustomThread1(Thread):
    # override the run function
	def run(self):
		print("retrieving from the queue")
		item = queue1.get()
		print(f'I got from the queue {item}')
		time.sleep(0.1)


if __name__ == '__main__':
	#source = "rtsp://root:mubea@10.65.68.2:8554/axis-media/media.amp"
	source = "rtsp://user:user@10.65.68.47/axis-media/media.amp"
	imgsz = 640
	stride = 32
	queue1 = queue.Queue()
	thread = CustomThread(target=CustomThread.capture())
	#thread2 = CustomThread1(target=CustomThread1.run,args=(queue1,))
	thread.start()
	#thread2.start()
	

