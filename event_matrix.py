import numpy as np
import cv2

class EventMatrix():
	def __init__(self, start_stop_times=[]):
		self.start_stop_times = start_stop_times

	def init_img(self, image):
		self.start_stop_times = self.obtain_start_stop_times_from_img(image)

	def obtain_start_stop_times_from_img(self, img):
		# Generate a list of start and stop times from a target image
		start_stop_times = []
		for r in range(img.shape[0]):
			start_stop_for_single_event = []

			nonzero_indexes = np.nonzero(img[r])[0]
			start = nonzero_indexes[0]
			for x in range(1, len(nonzero_indexes)):
				if nonzero_indexes[x] - nonzero_indexes[x-1] > 1:
					start_stop_for_single_event.append((start, nonzero_indexes[x-1]))
					start = nonzero_indexes[x]
			start_stop_for_single_event.append((start, nonzero_indexes[-1]))

			all_start_stop.append(start_stop_for_single_event)
		
		return start_stop_times

	def getImg(self, max_time = -1, write_filename=''):
		num_events = len(self.start_stop_times)

		img = np.zeros((num_events, max_time))
		for event in range(num_events):
			for t in self.start_stop_times[event]:
				if(t[0] >= 0):
					img[event, t[0]:t[1]] = 1

		if(len(write_filename) > 0 ):
			print("write: ", write_filename)

			img_copy = np.copy(img)
			img -= 1
			for n in range(0, img.shape[1], 10):
				img[:,n] *= 0.5
			img = np.stack((img,img,img), axis = 2)
			img *= -255

			#resize the image
			sizer = 6
			img = cv2.resize(img, None,fx=sizer, fy=sizer, interpolation=cv2.INTER_NEAREST)

			#add black spacers
			line = np.zeros_like(img.shape[1])
			for x in range(img.shape[0], 0, -sizer):
				img = np.insert(img, x, line, axis=0)
			for x in range(img.shape[1], 0, -sizer):
				img = np.insert(img, x, line, axis=1)

			#write the file
			cv2.imwrite(write_filename, img)
			img = img_copy

		return img

	def getStartStopTimes(self):
		return self.start_stop_times

