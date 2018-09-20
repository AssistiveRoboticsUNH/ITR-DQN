import numpy as np
import threading, math

THRESHOLDING_OPTIONS = ["mean", "histogram", "entropy", "norm"]

class CountDownLatch(object):
	#A Count down latch class to assist in multithreading
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

def threshold_input(activations, out, latch, thresholding_method):
	'''
	Perform the specified thresholding method on the activations. 
		-activations: a numpy array indicating relative expression of the 
				activation map for a given filter
		-out: a list to place the thresholded output in
		-latch: a count down latch to assist multithreading
		-thresholding_method: a string either "mean", "histogram", "entropy",
				or "norm"that designates the type of thresholding to perform
	'''

	# Choose the maximum expresion that occurs spatially in each frame for each 
	#filter 
	activations = np.max(activations, axis=1)
	returned_values = None

	if(thresholding_method == "norm"):
		# Perform scaling/normalization
		activations -= activations.min()
		if(activations.max() != 0):
			activations /= activations.max()
		else:
			activations = list(np.zeros_like(activations))
		returned_values = activations

	else:
		threshold = 0.0
		if(thresholding_method == "mean"):
			# Perform mean thresholding
			threshold = np.mean(activations)

		else:
			# Set up histograms for iterative methods
			hist, bin_edges = np.histogram(activations, bins, normed=False)
			pmf = [float(i)/np.sum(hist) for i in hist]
			
			metric, min_val = None, 0

			if(thresholding_method == "histogram"):
				# Perform histogram-based thresholding
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

					if(metric == None or eta_t < metric):
						metric = eta_t
						min_val = i

			elif(thresholding_method == "entropy"):
				# Perform entropy-based thresholding
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

						if(mean_a != 0 and mean_b != 0):

							eta_t = - (moment1a * math.log(mean_a)) - (moment1b * math.log(mean_b))
							if(metric == None or eta_t < metric):
								metric = eta_t
								min_val = i

			# select the threshold number that performed the best
			threshold = bin_edges[min_val]
			returned_values = activations > threshold

	out.append(returned_values)
	latch.count_down()

def thresholding(activation_map, data_ratio, thresholding_method="basic"):
	'''
	Perform the specified thresholding method on the entire activation map provided. 
		-activation_map: a 4D numpy array containing the output of a 3D-CNN convolution
		-data_ratio: a float indicating the amount of usefule (un-padded) frames
				in the original input, the rest can be ignored
		-thresholding_method: a string either "mean", "histogram", "entropy", or "norm"
				that designates the type of thresholding to perform
	'''

	# asserts to make sure input is correctly formed
	assert len(activation_map.shape) == 4, "input to 'feature_to_event' must be 4D matrix, input has "+str(len(activation_map.shape))+" dimensions"
	assert data_ratio > 0.0 and data_ratio <= 1.0, "Data ratio parmater must be a float 0 < x <= 1, is :" + str(data_ratio)
	assert thresholding_method in THRESHOLDING_OPTIONS, "Thresh_method must be in "+str(THRESHOLDING_OPTIONS)+", is: "+ str(thresholding_method)

	num_events = activation_map.shape[-1]
	initial_length = activation_map.shape[0]

	#only perform thresholding on the un-padded region of the input
	activation_map = np.array(activation_map)[:int(data_ratio*activation_map.shape[0])]

	#set up a list to contain the identified start and stop times
	thresholded_activations = []
	for i in range(num_events):
		thresholded_activations.append([])
	
	#setup multithread processes to perform the threading operation
	latch = CountDownLatch(num_events)

	list_of_threads = []
	for i in range(num_events):
		# isolate the activations for filter 'i' and perform the thresholding for 
		# this feature in its own thread
		activations = np.reshape(activation_map[...,i], (activation_map.shape[0], -1)) 

		t = threading.Thread(target = threshold_input, args = (activations, thresholded_activations[i], latch, thresholding_method, ))

		list_of_threads.append(t)
		t.start()
		
	# wait for all threads to finish
	latch.await()

	# pad and resize array for use in the ITR network
	thresholded_activations = np.array(thresholded_activations).squeeze().astype(np.int64)
	thresholded_activations = np.pad(thresholded_activations, 
									((0,0), (0,initial_length-thresholded_activations.shape[-1])), 
									'constant', 
									constant_values=(0,0))

	return thresholded_activations 