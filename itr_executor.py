'''
model_executor.py
Madison Clark-Turner
04/30/2018

A framework for executing different 3D-CNN models. The purpose of this file is to make the
file access simpler and provide an easily modifiable platform for furture code.
'''
import numpy as np
import tensorflow as tf 

# model structure
import c3d_vars
import cnn2d_vars

# thresholding
from event_matrix import EventMatrix
from feature_to_event import feature_to_event_category, norm_event#feature_to_event

# file io
import os
from os.path import isfile, join


# helper methods
import sys
import time
import cv2

RES_FILE = "../../C3D-tensorflow/sports1m_finetuning_ucf101.model"
#INP_FILE = "../../tfrecords_g/"
#VAL_FILE = "../../tfrecords_g_validation/"
INP_FILE = "../../tfrecords_corrected/"
VAL_FILE = "../../tfrecords_test_corrected/"#"../../tfrecords_validation2/"#

compute_server = False
if(compute_server):
	from input_pipeline_3dstr import input_pipeline
	RES_FILE = "sports1m_finetuning_ucf101.model"
	INP_FILE = "tfrecords_g/"
	VAL_FILE = "tfrecords_g_validation/"
else:
	from cnn3d.input_pipeline_3dstr import input_pipeline


class ITRExecutor:
	'''
		training_dir - String; the directory of the training tf_records
		training_dir - String; the directory of the testing tf_records
	'''
	def __init__(self, training_dir=None, train_len=-1, testing_dir=None, c3d_depth=0, test_len=-1, restore_full_chkpt='', thresh_method="basic", dataset_limit=0):

		#Properties

		batch_size = 1
		self.learning_rate = 1e-4
		self.c3d_depth = c3d_depth
		self.thresh_method = thresh_method

		#Setup Model

		self.restore_filename = RES_FILE

		self.c3d_input_ph = c3d_vars.get_input_placeholder(batch_size)
		self.c3d_w, self.c3d_b = c3d_vars.get_variables()
		self.c3d_activation_map = c3d_vars.generate_feature_activations(self.c3d_input_ph, self.c3d_w, self.c3d_b, c3d_depth)
		c3d_varlist = list( set(self.c3d_w.values() + self.c3d_b.values()) )


		self.cnn2d_input_ph = cnn2d_vars.get_input_placeholder(batch_size, c3d_depth)
		self.cnn2d_output_ph = cnn2d_vars.get_output_placeholder(batch_size)
		self.cnn2d_w, self.cnn2d_b = cnn2d_vars.get_variables(c3d_depth)
		self.generate_classification = cnn2d_vars.generate_classification(self.cnn2d_input_ph, self.cnn2d_w, self.cnn2d_b, c3d_depth)
		cnn2d_varlist = list( set(self.cnn2d_w.values() + self.cnn2d_b.values()) )

		self.optimize = self.optimizer()
		self.eval = self.classify()
		#Load files

		self.training_records, train_iter = self.open_tfrecord_dir(training_dir, limit_dataset=dataset_limit)
		self.train_iter = train_iter if train_len < 0 else train_len
		
		self.testing_records, test_iter = self.open_tfrecord_dir(testing_dir, randomize=False)
		self.test_iter =  test_iter if test_len < 0 else test_len
		print("self.train_len:", train_len, "self.test_iter:", self.test_iter)
	
		self.__finalized = False

		#Session

		self.sess = tf.InteractiveSession()

		self.saver = tf.train.Saver(c3d_varlist)
		if(restore_full_chkpt != ''):
			self.restore_filename = restore_full_chkpt
			all_vars = list( set(c3d_varlist.values() + cnn2d_varlist.values()) )
			self.saver = tf.train.Saver(all_vars)
	
	#-----------------------
	# Model Functions 
	#-----------------------

	def saveModel(self, name="model.ckpt", save_dir="", step=0):
		# create a checkpoint
		path = self.saver.save(self.sess, save_dir+'/'+name, global_step=step)

	def run_global_initalizers(self):
		# only run when training and before finalize()
		self.sess.run(tf.global_variables_initializer())
		
	def finalize(self):
		# complete model setup (should be executed prior to run())
		self.sess.run(tf.local_variables_initializer())

		#restore model values
		self.saver.restore(self.sess, self.restore_filename)

		# ensure no additional changes are made to the model
		self.sess.graph.finalize()

		# start queue runners in order to read ipnut files
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		# used for generation of meta-data
		self.run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		self.run_metadata=tf.RunMetadata()

		self.__finalized = True

	#-----------------------
	# File IO
	#-----------------------

	def open_tfrecord_dir(self, path, randomize=False, limit_dataset=0):
		# generate list of filenames
		all_files = [path+f for f in os.listdir(path) if isfile(join(path, f))]
		'''
		all_files = []
		for x in range(2):
			filenames = []
			if(x == 0):
				filenames = [f for f in os.listdir(path) if isfile(join(path, f)) and f.find("g") >= 0]
			else:
				filenames = [f for f in os.listdir(path) if isfile(join(path, f)) and f.find("g") < 0]
			filenames = [path+x for x in filenames ]
		
			if(limit_dataset > 0):
				filenames = filenames[:limit_dataset]
			all_files += filenames[:]
		'''
		all_files.sort()
		#print(all_files)

		if(randomize):
			random.shuffle(all_files)

		return input_pipeline(all_files, True, [1,1,1], randomize=randomize), len(all_files)

	def generate_model_input(self, data_source):
		# function returns output of a parsed tf_record call from the specified data_source (ie. self.training_records)
		input_values = [self.sess.run(data_source)][0]

		# read data into numpy arrays
		sequence_data = [None]*3
		labels = [None]*3
		data_ratio = 0.0
		for x in range(cnn2d_vars.MODAL_INPUTS):
			
			sequence_data[x] = input_values.pop(0)
			
			if(x == 0):
				# get part of data that contains information of interest
				data_ratio = (float(sequence_data[0].shape[1])/ \
						(cnn2d_vars.IC_DTYPE[x]["cmp_h"]*cnn2d_vars.IC_DTYPE[x]["cmp_w"]*cnn2d_vars.IC_DTYPE[x]["num_c"])) / \
						cnn2d_vars.NUM_FRAMES_PER_CLIP

				buffer_len = (cnn2d_vars.NUM_FRAMES_PER_CLIP * \
								cnn2d_vars.IC_DTYPE[x]["cmp_h"] * \
								cnn2d_vars.IC_DTYPE[x]["cmp_h"] * \
								cnn2d_vars.IC_DTYPE[x]["num_c"])-sequence_data[0].shape[1]
				
				sequence_data[x] = np.pad(sequence_data[x], 
											((0,0), (0,buffer_len)), 
											'constant', 
											constant_values=(0,0))
				sequence_data[x] = sequence_data[x].reshape((-1, 
											cnn2d_vars.IC_DTYPE[x]["cmp_h"], 
											cnn2d_vars.IC_DTYPE[x]["cmp_w"], 
											cnn2d_vars.IC_DTYPE[x]["num_c"]))
				
				output_shape = (len(sequence_data[x]), 
								c3d_vars.CROP_SIZE, 
								c3d_vars.CROP_SIZE, 
								cnn2d_vars.IC_DTYPE[x]["num_c"])

				# need to account for Opt Flow and Aud which have only 1 channel
				if(output_shape[3] == 1):
					output_shape = (len(sequence_data[x]), c3d_vars.CROP_SIZE, c3d_vars.CROP_SIZE)

				# resize image to work with C3D and account for potential absent channels
				h,w = sequence_data[x].shape[1], sequence_data[x].shape[2]
				img_stack_sm = np.zeros(output_shape)
				for idx in range(len(sequence_data[x])):
					img = sequence_data[x][idx, :, :]
					img_sm = cv2.resize(img, None,fx=float(c3d_vars.CROP_SIZE)/w, fy=float(c3d_vars.CROP_SIZE)/h, interpolation=cv2.INTER_CUBIC)
					img_stack_sm[idx, :, :] = img_sm

				sequence_data[x] = img_stack_sm

			labels[x] = input_values.pop(0)[0]
			#labels[x] = np.unravel_index(np.argmax(labels[x], axis=None), labels[x].shape)
		#print("labels: ", labels)
		p_t = np.expand_dims(input_values[-1], axis=0)
		example_id = input_values[-1]

		vals = {}
		vals[self.c3d_input_ph] = np.expand_dims(sequence_data[0], axis=0)

		return vals, data_ratio, labels[1], example_id

	def threshold_activation_map(self, activation_map, data_ratio, labels, example_id, write_filename=''):
		# function returns output of a parsed tf_record call from the specified data_source (ie. self.training_records)
		
		thresholded_map = None
		if (self.thresh_method != "norm"):
			start_stop = feature_to_event_category(activation_map[0], data_ratio, self.thresh_method)
			#start_stop = feature_to_event(activation_map[0], subdivide=1, threshold=0.5)
			em = EventMatrix(start_stop)
			thresholded_map = em.getImg(cnn2d_vars.NUM_MOMENT[self.c3d_depth], write_filename=write_filename)
		else:
			thresholded_map = norm_event(activation_map[0], data_ratio)

		vals = {}
		vals[self.cnn2d_input_ph] = np.reshape(thresholded_map, [1, 
										thresholded_map.shape[0],
										thresholded_map.shape[1],
										1])
		vals[self.cnn2d_output_ph] = labels

		return vals

		
	
	#-----------------------
	# Classification Functions 
	#-----------------------
	
	# return the action with the highest q-value
	def classify(self):
		return tf.argmax(self.generate_classification,1)
	
	#-----------------------
	# Optimization Functions 
	#-----------------------

	# generate the cross entropy 
	def cross_entropy(self):
		return tf.nn.softmax_cross_entropy_with_logits(
				labels=self.cnn2d_output_ph, logits=self.generate_classification)

	# optimze/train the network 
	def optimizer(self):
		return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
			self.cross_entropy())

	#-----------------------
	# Run 
	#-----------------------

	def run(self):
		assert self.__finalized, "Must execute finalize function before calling run"
		return self.runWrapper()

	def runWrapper(self):
		assert True,"run function must be defined"
		# modify for each new application

		print("begin training")

		for iteration in range(self.train_iter):	
			self.train()

			# evaluate system accuracy on test dataset once every 1000 iterations
			if(iteration %500 == 0 and True): #1000
				t_s = time.time()

				confusion_matrix = self.evaluate(write_img=(iteration==0))

				print(str(iteration))
				print("acc: "+str(self.get_accuracy(confusion_matrix)))

				# save the model every 10K iterations
				if(iteration % 100 == 0):
					self.saveModel(save_dir="itr", step=iteration)
			
		confusion_matrix = self.evaluate()
		print(str(iteration))
		print("acc: "+str(self.get_accuracy(confusion_matrix)))

		# save final model to chekpoint file
		self.saveModel(save_dir="itr_final", step=iteration)

	def train(self):
		vals, data_ratio, labels, example_id = self.generate_model_input(self.training_records)
		activation_map = self.sess.run(self.c3d_activation_map, feed_dict=vals)
		#print(np.asarray(activation_map).shape)

		vals = self.threshold_activation_map(activation_map, data_ratio, labels, example_id)
		self.sess.run(self.optimize, feed_dict=vals) 

	def evaluate(self, verbose=False, write_img=False):
		# evaluate the model using the validation dataset
		confusion_matrix = np.zeros((cnn2d_vars.NUM_LABEL, cnn2d_vars.NUM_LABEL))	

		if(verbose):
			print "%-*s  %s\t%s\t%s\t%s" % (20,"ID","Labl", "Pred", "Score", "Value")

		gest_obs = False
		print("testiter:", self.test_iter)
		for i in range(self.test_iter):
			
			vals, data_ratio, labels, example_id = self.generate_model_input(self.testing_records)
			activation_map = self.sess.run(self.c3d_activation_map, feed_dict=vals)
			
			write_filename = ''
			#print("labels:", labels)
			#if(write_img and not gest_obs and labels[0][1]):
			#	write_filename = "pics/"+self.thresh_method+"_d"+str(self.c3d_depth)+".png"
				#np.save("pics/"+self.thresh_method+"_d"+str(self.c3d_depth)+"_.npy", activation_map)
			
			vals = self.threshold_activation_map(activation_map, data_ratio, labels, example_id, write_filename)

			if(write_img and not gest_obs and labels[0][1]):
				write_filename = "pics/"+self.thresh_method+"_d"+str(self.c3d_depth)+"_.npy"
				np.save("pics/"+self.thresh_method+"_d"+str(self.c3d_depth)+"_.npy", vals[self.cnn2d_input_ph])
				print("file written to: "+write_filename)

				gest_obs = True

			mcp = self.sess.run(self.eval, feed_dict=vals)

			true_label = np.unravel_index(np.argmax(labels[0], axis=None), labels[0].shape)
			pred_label = mcp[0]
			print(i, example_id, true_label, pred_label)
			confusion_matrix[true_label][pred_label] += 1
				
		return confusion_matrix

	def get_accuracy(self, confusion_matrix):
		correct = 0
		for n in range(cnn2d_vars.NUM_LABEL):
			correct += confusion_matrix[n][n]
		return correct/float(self.test_iter)

if __name__ == '__main__':
	# dataset_size test


	train_len = 3000
	
	if(compute_server):

		#model accuracy test

		# dataset_size test
		for thresh_method in ["norm"]:#["basic", "histogram", "entropy"]:
			for depth in range(0, 5):
				print("thresh_method:", thresh_method, "depth:", depth)
				itr_exec = ITRExecutor(
								train_len=train_len, 
								training_dir=INP_FILE, 
								testing_dir=VAL_FILE, 
								c3d_depth=depth,
								thresh_method=thresh_method,
								dataset_limit=80)
				itr_exec.run_global_initalizers()
				itr_exec.finalize()
				itr_exec.run()
				tf.reset_default_graph()
	else:
		itr_exec = ITRExecutor(
								train_len=train_len, 
								training_dir=INP_FILE, 
								testing_dir=VAL_FILE, 
								c3d_depth=0,
								thresh_method="norm")
		itr_exec.run_global_initalizers()
		itr_exec.finalize()
		itr_exec.run()
		tf.reset_default_graph()
		'''
		#for thresh_method in ["norm"]:
		for thresh_method in ["basic", "histogram", "entropy", "norm"]:
			#for n in range(30, 100, 10):
			for depth in range(0, 5):
				#print("thresh_method:", thresh_method, "dataset_limit:",n , "depth:", depth)
				print("thresh_method:", thresh_method, "depth:", depth)
				
				itr_exec = ITRExecutor(
								train_len=train_len, 
								training_dir=INP_FILE, 
								testing_dir=VAL_FILE, 
								c3d_depth=depth,
								thresh_method=thresh_method,
								dataset_limit=5)
				itr_exec.run_global_initalizers()
				itr_exec.finalize()
				itr_exec.run()
				tf.reset_default_graph()
		'''

