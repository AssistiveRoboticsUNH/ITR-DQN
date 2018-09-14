'''
feature_to_event.py
Madison Clark-Turner
12/11/2017

Converts the output of model_3d (feature activations) into the input for
model_str (events)
'''
import numpy as np
#from skimage.filters import threshold_otsu, threshold_adaptive
#from skimage.util import view_as_blocks
import threading
import time

from event_matrix import EventMatrix

from scipy import signal
import math

class CountDownLatch(object):
	def __init__(self, count=1):
		self.count = count
		self.lock = threading.Condition()

	def count_down(self):
		self.lock.acquire()
		self.count -= 1
		if self.count <= 0:
			self.lock.notifyAll()
		self.lock.release()

	def await(self):
		self.lock.acquire()
		while self.count > 0:
			self.lock.wait()
		self.lock.release()

def threshold_input(activations, threshold, start_stop_times):
	expression_above_threshold = [activations > threshold]
	expression_indicies = np.nonzero(expression_above_threshold)[1]

	#s_t = time.time()
	if(len(expression_indicies) > 0):
		start_time = expression_indicies[0]
		for j in range(1, len(expression_indicies)):
			if (expression_indicies[j] - expression_indicies[j-1] > 1):
				start_stop_times.append((start_time, expression_indicies[j-1]))
				start_time = expression_indicies[j]

		if(start_time != expression_indicies[-1]):
			start_stop_times.append((start_time, expression_indicies[-1]))

def histogram_event(activations, start_stop_times, latch, bins):
	activations = np.max(activations, axis=1)
	hist, bin_edges = np.histogram(activations, bins, normed=False)
	pmf = [float(i)/np.sum(hist) for i in hist]


	#print("hist:", hist)
	'''
	print("a:", activations)
	print("bin_edges:", bin_edges)
	print("hist:", hist)
	print("pmf:", pmf)
	print('')
	'''
	
	metric, min_val = None, 0
	for i in range (1,len(hist)):

		pmf_t = np.sum(pmf[:i])

		chunk = np.copy(hist[:i]) * range(1, i+1)
		b1 = np.sum(chunk) / pmf_t

		chunk = np.copy(hist[i:]) * range(i+1, len(hist)+1)
		b2 = np.sum(chunk) * (1 - pmf_t)
		sum_b1 = [(b1 - j)**2 for j in range(0, i)]
		sum_b1 = np.sum(sum_b1)

		sum_b2 = [(b2 - j)**2 for j in range(i+1, len(hist))]
		sum_b2 = np.sum(sum_b2)

		eta_t = sum_b1 + sum_b2
		#print(i, eta_t)

		if(metric == None or eta_t < metric):
			metric = eta_t
			min_val = i

	threshold = bin_edges[min_val]
	#print("threshold: ", threshold)
	#print("min_val: ", min_val)

	threshold_input(activations, threshold, start_stop_times)

	latch.count_down()

def entropy_event(activations, start_stop_times, latch, bins):
	activations = np.max(activations, axis=1)
	hist, bin_edges = np.histogram(activations, bins)

	pmf = [float(i)/np.sum(hist) for i in hist]
	#print("hist:", hist)

	'''
	print("a:", activations)
	print("bin_edges:", bin_edges)
	print("hist:", hist)
	print('')
	'''

	#entropy division
	metric, min_val = None, 0
	for i in range (1,len(hist)):

		moment0a = np.sum(hist[:i])
		chunk = np.copy(hist[:i]) * range(1, i+1)
		moment1a = np.sum(chunk)

		moment0b = np.sum(hist[i:])
		chunk = np.copy(hist[i:]) * range(i+1, len(hist)+1)
		moment1b = np.sum(chunk)

		if(moment0a != 0 and moment0b != 0):

			mean_a = moment1a/moment0a
			mean_b = moment1b/moment0b

			#print("mean_a:", mean_a, "mean_b:", mean_b)
			if(mean_a != 0 and mean_b != 0):

				eta_t = - (moment1a * math.log(mean_a)) - (moment1b * math.log(mean_b))
				if(metric == None or eta_t < metric):
					metric = eta_t
					min_val = i

	threshold = bin_edges[min_val]

	#print("threshold: ", threshold)
	#print("min_val: ", min_val)

	threshold_input(activations, threshold, start_stop_times)

	latch.count_down()

def threshold_event_category(activations, summary, start_stop_times, latch, stdev):
	activations = np.max(activations, axis=1)
	if (summary == 'mean'):
		activations = np.mean(activations, axis=1)
	
	threshold = np.mean(activations) + np.var(activations)*(stdev-0.5)*2
	#print("threshold: ", threshold)

	threshold_input(activations, threshold, start_stop_times)
	
	latch.count_down()

def feature_to_event_category(activation_map, data_ratio, thresh_method="basic", summary="max", stdev= 0.5, window=10, num_bins=10):
	assert summary in ['mean', 'max', 'separate']

	#normalize the input
	s_t = time.time()
	activation_map = np.array(activation_map)[:int(data_ratio*activation_map.shape[0])]
	

	assert len(activation_map.shape) == 4, "input to 'feature_to_event' must be 4D matrix, input has "+str(len(activation_map.shape))+" dimensions"
	#activation_map /= activation_map.max()

	thresh_options = ["basic", "histogram", "entropy", "norm"]
	assert thresh_method in thresh_options, "Thresh_method must be in "+str(thresh_method)+", is: "+ str(thresh_method)

	if(False):
		#print(activation_map)
		
		print("max: ", np.max(activation_map))
		print("min: ", np.min(activation_map))
		print("median: ", np.median(activation_map))
		print("mean: ", np.mean(activation_map))
		print("var: ", np.var(activation_map))

	num_events = activation_map.shape[-1]

	event_times = []
	for i in range(num_events):
		event_times.append([])

	#print("start: ", time.time()-s_t)

	# for each filter get the start and stop times of the feature
	
	latch = CountDownLatch(num_events)

	reshape_t, mod_t_1, mod_t, thresh_t, index_t = 0, 0, 0, 0, 0
	all_threads = []
	s_t = time.time()
	for i in range(num_events):
		# flatten spatila dimension in order to identify max for each time segment
		activations = np.reshape(activation_map[...,i], (activation_map.shape[0], -1)) 
		
		#t = threading.Thread(target = threshold_event_category, args = (activations, summary, event_times[i], latch, stdev, ))
		#if(summary == "separate"):



		if(thresh_method == "basic"):
			t = threading.Thread(target = threshold_event_category, args = (activations, summary, event_times[i], latch, stdev, ))
		elif(thresh_method == "histogram"):
			t = threading.Thread(target = histogram_event, args = (activations, event_times[i], latch, num_bins))
		elif(thresh_method == "entropy"):
			t = threading.Thread(target = entropy_event, args = (activations, event_times[i], latch, num_bins))
		
		


		#	t = threading.Thread(target = window_thresholding_event, args = (activations, summary, event_times[i], latch, window, ))
		#t = threading.Thread(target = histogram_event, args = (activations, event_times[i], latch, 10))
		
		all_threads.append(t)
		t.start()
		
		
	index_t += time.time()-s_t
	latch.await()
	
	return event_times 

def norm_feature(activations, return_loc, latch):
	activations = np.max(activations, axis=1)
	
	if(activations.max() != 0):
		activations = [(a-activations.min())/activations.max() for a in activations]
	else:
		activations = list(np.zeros_like(activations))
	return_loc.append(activations)
	
	latch.count_down()


def norm_event(activation_map, data_ratio):
	#normalize the input
	original_length = activation_map.shape[0]
	actual_length = int(data_ratio*activation_map.shape[0])
	activation_map = np.array(activation_map)[:actual_length]
	
	assert len(activation_map.shape) == 4, "input to 'feature_to_event' must be 4D matrix, input has "+str(len(activation_map.shape))+" dimensions"

	num_events = activation_map.shape[-1]

	event_times = []
	for i in range(num_events):
		event_times.append([])

	latch = CountDownLatch(num_events)

	all_threads = []
	for i in range(num_events):
		activations = np.reshape(activation_map[...,i], (activation_map.shape[0], -1)) 
		
		t = threading.Thread(target = norm_feature, args = (activations, event_times[i], latch, ))
		
		all_threads.append(t)
		t.start()
		
	latch.await()

	event_times = np.asarray([a[0] for a in event_times])
	event_times = np.pad(event_times, ((0,0), (0,original_length - actual_length)), 'constant', constant_values=(0, 0))

	#print(event_times.shape, data_ratio, original_length)

	return event_times 


if __name__ == '__main__':

	NUM_MOMENT = [256,256, 128,64,32]

	'''
	for depth in range(1):
	
		#win_len = int(NUM_MOMENT[depth] * window)
		#print("WIN_LEN:", win_len)
		data = np.load("oldpics/thresh_d"+str(depth)+"_max_0.5.npy")[0]
		print(data.shape)
		#event_times = norm_event(data, 0.5)
		event_times = feature_to_event_category(data, 0.5, thresh_method="basic")

		print(event_times.shape)
		
	'''
	
	for thresh_method in ["basic", "histogram", "entropy"]:
		for depth in range(5):
		
			#win_len = int(NUM_MOMENT[depth] * window)
			#print("WIN_LEN:", win_len)
			data = np.load("pics/thresh_d"+str(depth)+"_max_0.5.npy")[0]
			print(data.shape)
			event_times = feature_to_event_category(data, 0.5, thresh_method=thresh_method)#, window=win_len)
			
			em = EventMatrix(event_times)
			em.getImg(NUM_MOMENT[depth], write_filename="pics/"+thresh_method+"_d"+str(depth)+".png")
			print('')
	
		
