#!/usr/bin/env python

'''
generate_tfrecords_from_rosbag_3dstr.py
Madison Clark-Turner
12/14/2017

Convert rosbags into TFrecords
'''

import tensorflow as tf
import numpy as np

#file IO
import rospy
import rosbag
import os
from os.path import isfile, join
from file_io_dqn import input_pipeline

#used for performing pre-processing steps on rostopics
from packager import DataPackager 

#used for viewing TFRecord contents
import cv2

#rostopic names
topic_names = [
	'/action_finished',
	'/nao_robot/camera/top/camera/image_raw',
	'/nao_robot/microphone/naoqi_microphone/audio_raw'
]

'''
Read contents of a Rosbag and store:
s  - the current observation
a  - action that followed s
s' - the subsequent observation
a' - the action that followed s'
p  - how many prompts had been delivered before s
'''

def make_sequence_example(
	img_raw, img_lab, 
	aud_raw, aud_lab,
	p_t, 
	first_action,
	example_id,
	second_action = -1,
	img_raw2=np.array([]),   
	aud_raw2=np.array([])):

	# The object we return
	ex = tf.train.SequenceExample()

	# ---- descriptive data ----

	ex.context.feature["length"].int64_list.value.append(img_raw.shape[1])

	len2 = 0
	if(len(img_raw2) > 0):
		len2 = img_raw2.shape[1]
	ex.context.feature["length2"].int64_list.value.append(len2)
	ex.context.feature["example_id"].bytes_list.value.append(example_id)

	# ---- label data ----

	ex.context.feature["p_t"].int64_list.value.append(p_t)
	ex.context.feature["total_lab"].int64_list.value.append(first_action)
	ex.context.feature["total_lab2"].int64_list.value.append(second_action)

	ex.context.feature["img_lab"].int64_list.value.append(img_lab)
	ex.context.feature["aud_lab"].int64_list.value.append(aud_lab)

	# ---- data sequences ----

	def load_array(example, name, data, dtype):
		fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
		print("newShape:", np.asarray(data).astype(dtype).shape)
		fl_data.append(np.asarray(data).astype(dtype).tostring())

	load_array(ex, "img_raw", img_raw, np.uint8)
	load_array(ex, "aud_raw", aud_raw, np.uint8)#float64

	load_array(ex, "img_raw2", img_raw2, np.uint8)
	load_array(ex, "aud_raw2", aud_raw2, np.uint8)#float64

	return ex

def gen_TFRecord_from_file(out_dir, out_filename, bag_filename, flip=False):
	packager = DataPackager(flip=flip)
	bag = rosbag.Bag(bag_filename)	

	output_filenames = []

	#######################
	##  Get Label Info   ##
	#######################

	example_id = out_filename

	file_end = bag_filename.find(".bag")
	label_code = bag_filename[file_end-5:file_end]
	print("")
	print("bag_filename: ", bag_filename)
	print("label_code:", label_code)

	img_lab, opt_lab, aud_lab = 0,0,0
	if("z" in example_id):
		img_lab = 1
	if("g" in example_id):
		opt_lab = 1
	if("a" in example_id):
		aud_lab = 1
	total_lab = (img_lab+opt_lab+aud_lab > 0)

	print(example_id)
	print(img_lab, opt_lab, aud_lab, ':', total_lab)

	end_file = ".tfrecord"
	if(flip):
		end_file = "_flip"+end_file

	#######################
	##     READ FILE     ##
	#######################

	p_t = 0

	stored_data = []
	for topic, msg, t in bag.read_messages(topics=topic_names):
		if(topic == topic_names[0]):
			
			last_action = str(msg.data)

			if(msg.data > 0):
				# perform data pre-processing steps
				packager.formatOutput()

				if(msg.data == 1):
					print("packager.getImgStack().shape: ", packager.getImgStack().shape)
					stored_data = {
						"img_raw": packager.getImgStack()[:], "img_lab": 0, 
						"aud_raw": packager.getAudStack()[:], "aud_lab": 0, 
						"p_t": p_t,
						"total_lab": int(last_action),
						"example_id": example_id}
					p_t += 1
					
				elif(msg.data > 1):
					break

			packager.reset()
		elif(topic == topic_names[1]):
			packager.imgCallback(msg)
		elif(topic == topic_names[2]):
			packager.audCallback(msg)

	if(p_t > 0):
		ex = make_sequence_example (
			img_raw=stored_data["img_raw"], img_lab=stored_data["img_lab"], 
			aud_raw=stored_data["aud_raw"], aud_lab=stored_data["aud_lab"], 
			p_t=stored_data["p_t"], 
			first_action=stored_data["total_lab"],
			example_id=stored_data["example_id"],
			img_raw2=packager.getImgStack(), 
			aud_raw2=packager.getAudStack(),
			second_action=int(last_action))
		output_filename = out_dir+out_filename+"_"+str(stored_data["total_lab"])+end_file
		output_filenames.append(output_filename)
		writer = tf.python_io.TFRecordWriter(output_filename)
		writer.write(ex.SerializeToString())
		writer.close()

	# generate TFRecord data
	ex = make_sequence_example (
		img_raw=packager.getImgStack(), img_lab=img_lab, 
		aud_raw=packager.getAudStack(), aud_lab=aud_lab, 
		p_t=p_t, 
		first_action=int(last_action),
		example_id=example_id)
	print("last_action:", msg.data, int(last_action))

	# write TFRecord data to file
	output_filename = out_dir+out_filename+"_"+last_action+end_file
	output_filenames.append(output_filename)
	writer = tf.python_io.TFRecordWriter(output_filename)
	writer.write(ex.SerializeToString())
	writer.close()

	packager.reset()
	bag.close()

	return output_filenames

if __name__ == '__main__':
	gen_single_file = False
	view_single_file = False
	process_all_files = True
	
	rospy.init_node('gen_tfrecord', anonymous=True)

#############################

	# USAGE: generate a single file and store it as a scrap.tfrecord; Used for Debugging

	bagfile = os.environ["HOME"] + \
		"/Documents/AssistiveRobotics/AutismAssistant/pomdpData/final_01/none1.bag"
	outdir = os.environ["HOME"]+'/'+"catkin_ws/src/cnn3d-str/scrap_records/"

	output_filenames= []

	if(gen_single_file):
		output_filenames = gen_TFRecord_from_file(out_dir=outdir, out_filename="scrap", bag_filename=bagfile, flip=False)

#############################
	
	# USAGE: read contents of scrap.tfrecord; Used for Debugging

	if(view_single_file):
		#Use for visualizing Data Types
		def show(data, d_type):

			data = np.reshape(data, (-1, d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))
			print(type(data), data.dtype)

			tout = []
			out = []
			for i in range(data.shape[0]):
				imf = data[i]

				limit_size = 64

				if(d_type["cmp_h"] > limit_size):
					mod = limit_size/float(d_type["cmp_h"])
					imf = cv2.resize(imf,None,fx=mod, fy=mod, interpolation = cv2.INTER_CUBIC)
					print(imf.shape)
				if(len(imf.shape) >=3 and imf.shape[2] == 2):
					
					imf = np.concatenate((imf, np.zeros((d_type["cmp_h"],d_type["cmp_w"],1))), axis=2)
					imf[..., 0] = imf[..., 1]
					imf[..., 2] = imf[..., 1]
					imf = imf.astype(np.uint8)

				if(i % 100 == 0 and i != 0 ):
					if(len(tout) == 0):
						tout = out.copy()
					else:
						tout = np.concatenate((tout, out), axis=0)
					
					out = []
				if(len(out) == 0):
					out = imf
				else:
					out = np.concatenate((out, imf), axis = 1)

			'''
			if(data.shape[0] % 10 != 0):

				fill = np.zeros((limit_size, limit_size*(10 - (data.shape[0] % 10)), d_type["num_c"]))
				fill.fill(255)
				out = np.concatenate((out, fill), axis = 1)
			'''

			return tout

		
		print("READING...")

		with tf.Session() as sess:

			
			#parse TFrecord
			print("output_filenames: ", output_filenames)
			#output_filenames = [ outdir+x for x in output_filenames]
			read_tensors = input_pipeline(output_filenames, randomize=False)

			sess.run(tf.local_variables_initializer())


			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# read data into numpy arrays
			
			#print(read_tensors)
			for i in range(len(output_filenames)): # alter number of iterations to the number of files
				print(i)
				input_values = sess.run(read_tensors)
				# assign data to placeholders
				
				

				output_values = [np.array(x) for x in input_values]

				'''
				extractFeature(sequence_parsed["img_raw"], cast=tf.uint8),  0
				extractFeature(sequence_parsed["aud_raw"], cast=tf.uint8),  1
				context_parsed["p_t"],                                      2
				tf.one_hot(context_parsed["total_lab"]-1, 3),               3
				context_parsed["example_id"],                               4

				extractFeature(sequence_parsed["img_raw2"], cast=tf.uint8), 5
				extractFeature(sequence_parsed["aud_raw2"], cast=tf.uint8), 6
				context_parsed["total_lab2"]                                7
				'''
				
				print("img_raw: ", output_values[0].shape)
				print("aud_raw: ", output_values[1].shape)
				
				print("img_raw2: ", output_values[5].shape)
				print("aud_raw2: ", output_values[6].shape)
				
				print("first_action: ", output_values[3])
				print("second_action: ", output_values[7])

				print("p_t: ", output_values[2])
				print("example_id: ", output_values[4])

				# display the contents of the optical flow file
				#img = show(output_values[4], AUD_DTYPE)
				#cv2.imshow("opt_data", img)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()
			coord.request_stop()
			coord.join(threads)
			print("close threads")
		
#############################

	# USAGE: write all rosbag demonstrations to TFRecords

	'''
	We assume that the file structure for the demonstartions is ordered as follows:

		-<demonstration_path>
			-<subject_id>_0
				-compliant
					-<demonstration_name_0>.bag
					-<demonstration_name_1>.bag
					-<demonstration_name_2>.bag
				-noncompliant
					-<demonstration_name_0>.bag

			-<subject_id>_1
			-<subject_id>_2

		-<tfrecord_output_directory>

	'''

	# setup input directory information
	demonstration_path = os.environ["HOME"]+\
			'/'+"Documents/AssistiveRobotics/AutismAssistant/pomdpData/"
	subject_id = "test"

	# setup output directory information
	tfrecord_output_directory = os.environ["HOME"]+'/'+"catkin_ws/src/cnn3d-str/tfrecords_extended/"#os.environ["HOME"]+'/'+"catkin_ws/src/cnn3d-str/tfrecords_test_corrected/"

	if(process_all_files):

		for i in range(9,10): # each unique subject
			subject_dir = demonstration_path + subject_id + '_'
			
			if(i < 10): # fix naming issues with leading 0s
				subject_dir += '0'
			subject_dir += str(i) + '/'

			#get list of demonstration file names
			filename_list = [subject_dir+f for 
					f in os.listdir(subject_dir) if isfile(join(subject_dir, f))]
			filename_list.sort()

			for f in filename_list:

				#get demonstration name for output file name
				tag = f
				while(tag.find("/") >= 0):
					tag = tag[tag.find("/")+1:]
				tag = tag[:-(len(".bag"))]
				new_name = subject_id+'_'+str(i)+'_'+tag

				# print files to make it clear process still running
				print(tag + "......." + new_name)

				gen_TFRecord_from_file(out_dir=tfrecord_output_directory, 
						out_filename=new_name, bag_filename=f, flip=False)

				gen_TFRecord_from_file(out_dir=tfrecord_output_directory, 
						out_filename=new_name, bag_filename=f, flip=True)