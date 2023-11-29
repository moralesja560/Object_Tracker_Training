import time
from threading import Thread
from random import randint

#https://superfastpython.com/threading-in-python/




# custom thread class
class CustomThread1(Thread):
    # override the run function
    def run(self):
        # block for a moment
        time.sleep(5)
        # display a message
        self.value = randint(1,9)
        print(f'This is the thread from the class 1 and the value is {self.value}')
    


# custom thread class
class CustomThread(Thread):
	def __init__(self,opt_arg):
		Thread.__init__(self)
		self.opt_arg = opt_arg
    # override the run function
	def run(self):
        # block for a moment
		time.sleep(5)
        # display a message
		print(f'This is the thread 2 and the value is {self.opt_arg}')

# create a thread
#target means that the function will be executed in another thread.
thread = CustomThread1()

#start will run the thread as soon as posible, not immediately
thread.start()
#join is to make the main thread explicitely wait for the thread to finish.
thread.join()
print(f"thread 1 finished and value is {thread.value}")
thread2 = CustomThread(opt_arg=thread.value)
thread2.start()
#join is to make the main thread explicitely wait for the thread to finish.
thread.join()
print("thread 1 finished")
# after the tread 1 has finished, we explicitely wait to the thread 2 to finish before putting the message.
thread2.join()
print("thread 2 finished")
