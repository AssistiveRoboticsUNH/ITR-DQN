import tensorflow as tf 
import numpy as np 

from file_io_dqn import read_files_in_dir, generate_model_input
import c3d
import itr_dqn_model_dqn as model_def
from thresholding_methods import thresholding


C3D_NETWORK_VARIABLE_FILE = "../../../C3D-tensorflow/sports1m_finetuning_ucf101.model"
CHKPT_NAME = ''#"itr_final/model.ckpt"
SAVE_NAME = 'itr_final/model.ckpt'
GAMMA = 0.9

def threshold_activation_map(ph_values, placeholders, information_values, activation_map, thresholding_approach="norm"):
	'''
	Applies the specified thresholding method to the provided activation map and 
	stores the result in the ITR input placeholder.
		- ph_values: a dictionary containing the placeholders
		-placeholders: the list of placeholders used by the network
		-information_values: a dictionary containing extraenous information about
				the input
		-activation_map: the C3D activation map
		-thresholding_approach: the thresholding method to use a string that is 
				either "mean", "histogram", "entropy", or "norm"
	'''

	# Apply thresholding to the provided activation map
	thresholded_map = thresholding(activation_map[0], information_values["data_ratio"], thresholding_approach)

	# multiply values by 255 to match the scale of values being used by the Audio network
	thresholded_map *= 255
	thresholded_map = np.reshape(thresholded_map,[1, thresholded_map.shape[0], thresholded_map.shape[1], 1])

	ph_values[placeholders["itr_in"]] = thresholded_map
	return ph_values

def obtain_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach, update_reward=False):
	'''
	Provides the training input for the ITR network by generating an IAD from the
	activation map of the C3D network. Outputs two dictionaries. The first contains
	the placeholders that will be used when evaluating the full ITR model. The second 
	contains information about the observation being read (ie. true labels, number of
	prompts, file name, etc). 
		-placeholders: the list of placeholders used by the network
		-tf_records: the TFRecord data source to read from
		-sess: the tensorflow Session
		-c3d_model: the c3d network model
	'''

	# Obtain activation amp from C3D network
	ph_values, info_values, sub_ph_values, sub_info_values = generate_model_input(placeholders, tf_records, sess)
	c3d_activation_map = sess.run(c3d_model, feed_dict=ph_values)

	ph_values = threshold_activation_map(ph_values, placeholders, info_values, c3d_activation_map, thresholding_approach)
	if(sub_info_values["label"] >= 0):
		sub_ph_values = threshold_activation_map(sub_ph_values, placeholders, sub_info_values, c3d_activation_map, thresholding_approach)

	# Perform thresholding on C3D activation map
	return ph_values, info_values, sub_ph_values, sub_info_values

def train(placeholders, tf_records, sess, c3d_model, thresholding_approach, optimizer, target_classifier):
	# Optimize variables in the rest of the network
	ph_values, info_values, sub_ph_values, sub_info_values = obtain_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach, update_reward=True)

	if(info_values["label"][0][0] > 0):
		# get the predicted value for the subsequent states
		target_prediction = sess.run(target_classifier, feed_dict=sub_ph_values)
		# update values with Q reward, reward for PMT is 0.1, other actions are terminal
		ph_values[placeholders["system_out"]][0][0] = 0.1 + target_prediction[0][0]*GAMMA
	#print("train_labels2:", ph_values[placeholders["system_out"]])

	sess.run(optimizer, feed_dict=ph_values) 

def identify_response(file_name):
	# Identify what responses the given file name were depicting
	tag = file_name
	while (tag.find('_') >= 0):
		tag = tag[tag.find('_')+1:]
	return tag[:-1]

def evaluate(placeholders, tf_records, sess, c3d_model, thresholding_approach, classifier, dataset_size, verbose=False):
	'''
	Evaluates the ITR model on the given dataset. Two outputs are generated: a 
	confusion matrix indicating the models correct and predicted guesses and a 
	dictionary containing the accuracy given the number of prompts executed and
	the response observed in the video.
	'''
	confusion_matrix = np.zeros((model_def.NUM_LABEL, model_def.NUM_LABEL))	
	responses = {}

	for i in range(dataset_size):
		# Generate predicted values
		ph_values, info_values, sub_ph_values, sub_info_values = obtain_IAD_input(placeholders, tf_records, sess, c3d_model, thresholding_approach, update_reward=False)
		predicted_action = sess.run(classifier, feed_dict=ph_values)[0]

		# Identify the correct action for the observation
		labels = info_values["label"][0]
		correct_action = np.unravel_index(np.argmax(labels, axis=None), labels.shape)

		# Update the confusion matrix
		confusion_matrix[correct_action][predicted_action] += 1


		# Determine the composition of the observation
		response_depicted = identify_response(info_values["example_id"][0])
		if(response_depicted not in responses):
			responses[response_depicted] = [[0,0],[0,0]]

		# Determine how many prompts have been delivered in the observation
		number_of_prompts = info_values["pt"]

		# update the response accuracy
		if correct_action == predicted_action:
			responses[response_depicted][number_of_prompts][0] += 1
		responses[response_depicted][number_of_prompts][1] += 1

		if(verbose):
			print(i, info_values["example_id"][0], number_of_prompts, correct_action, predicted_action)

	return confusion_matrix, responses

def get_accuracy(confusion_matrix):
	# Calculate the accuracy for a given confusion matrix
	correct = 0
	for n in range(confusion_matrix.shape[0]):
		correct += confusion_matrix[n][n]
	return correct/float(np.sum(confusion_matrix))

def run_model(
	num_train_iterations=10, 
	c3d_depth=0, 
	thresholding_approach="norm",
	training_dir='', 	
	training_dir_dataset_limit=0,
	validate_dir='', 
	testing_dir='',
	train_print_freq=0,
	validation_freq=0,
	save_freq=0,
	variable_update_freq=0):

	# ----------  setup variables ------------

	# setup variables
	placeholders = model_def.get_placeholders(c3d_depth=c3d_depth)
	weights_c3d, biases_c3d = c3d.get_variables()
	c3d_variable_names = list( set(weights_c3d.values() + biases_c3d.values()) )
	c3d_model = c3d.generate_activation_map(placeholders["c3d_in"], weights_c3d, biases_c3d, depth=c3d_depth)

	#define Q
	with tf.variable_scope('main'):
		weights_main, biases_main = model_def.get_variables(c3d_depth=c3d_depth)
		model = model_def.get_predicted_values(placeholders, weights_main, biases_main, c3d_depth=c3d_depth)
		optimizer = model_def.optimizer(placeholders, model, alpha=1e-3)
		classifier = model_def.classifier(model)
	variable_name_dict = model_def.list_variables(weights_main, biases_main)

	#define Q_hat
	with tf.variable_scope('target'):
		weights_target, biases_target = model_def.get_variables(c3d_depth=c3d_depth)
		model_target = model_def.get_predicted_values(placeholders, weights_target, biases_target, c3d_depth=c3d_depth)

	with tf.Session() as sess:

		# ----------  file I/O ------------

		# define files for training/testing

		training_records, testing_records, validate_records = None, None, None
		test_iter, valid_iter = 0, 0

		if(training_dir != ''):
			training_records, _ = read_files_in_dir(training_dir, randomize=True, limit_dataset=training_dir_dataset_limit, recursive=True)
		
		if(testing_dir != ''):
			testing_records, test_iter = read_files_in_dir(testing_dir, randomize=False, recursive=True)
		
		if(validate_dir != ''):
			validate_records, valid_iter = read_files_in_dir(validate_dir, randomize=False, recursive=False)

		# ----------  restore variables (update) ------------

		var_list = list( (variable_name_dict["itr"] + \
						variable_name_dict["aud"] + \
						variable_name_dict["system"]) )
		
		var_dict = {}
		for v in [v.name for v in var_list]:
			with tf.variable_scope("target", reuse=True):
				var_dict[v[:-2]] = tf.get_variable(v[v.find('/')+1:-2])
		
		
		restore_filename = C3D_NETWORK_VARIABLE_FILE

		if(CHKPT_NAME != ''):
			print("restoring checkpoint from :"+CHKPT_NAME)
			restore_filename = CHKPT_NAME
			
		# ----------  initalize variables ------------

		# setup variables
		if(train):
			sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		#initialize C3D network variables
		saver = tf.train.Saver( variable_name_dict["c3d"] )
		saver.restore(sess, restore_filename)

		#initialize other variables
		if(CHKPT_NAME != ''):
			saver = tf.train.Saver( var_list )
			print("restoring variables from "+CHKPT_NAME)
			saver.restore(sess, restore_filename)

		# ----------  finalize model ------------

		# ensure no additional changes are made to the model
		#sess.graph.finalize()

		# start queue runners in order to read ipnut files
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		# ----------  train network ------------

		for iteration in range(num_train_iterations):	

			# use target to get expected reward
			# apply discount reward
			# train actual network

			# update target avery 1000 iterations 




			train(placeholders, training_records, sess, c3d_model, thresholding_approach, optimizer, model_target)
			
			if(train_print_freq > 0 and iteration % train_print_freq == 0):
				print(iteration)
			
			if(validation_freq > 0 and iteration % validation_freq == 0): 
				# test the system on the validation dataset	
				ph_values, info_values, sub_ph_values, sub_info_values = obtain_IAD_input(placeholders, training_records, sess, c3d_model, thresholding_approach)
				print(sess.run(model, feed_dict=ph_values) )			
				#confusion_matrix, responses = evaluate(placeholders, validate_records, sess, c3d_model, thresholding_approach, classifier, valid_iter, verbose=False)
				#print("VAL "+str(iteration)+" accuracy: "+str(get_accuracy(confusion_matrix))+'\n')

			if(iteration > 0 and save_freq > 0 and iteration % save_freq == 0):
				# save the model to file
				saver.save(sess, SAVE_NAME)
				#pass

			#if(variable_update_freq > 0 and iteration % variable_update_freq == 0):
			if(variable_update_freq > 0 and iteration % variable_update_freq == 0):
				#update variables in the target network
				print("updating target network")
				
				if(CHKPT_NAME != ''):
					restore_filename = SAVE_NAME
					
					saver = tf.train.Saver( var_dict )
					print("pre rest, vars: ", sess.run(weights_target["system"]["W_1"]))
					saver.restore(sess, restore_filename)
					print("post rest, vars: ", sess.run(weights_target["system"]["W_1"]))
					saver = tf.train.Saver( var_list )
				

			
		# ----------  test network ------------

		# test the system on the testing dataset
		confusion_matrix, responses = evaluate(placeholders, testing_records, sess, c3d_model, thresholding_approach, classifier, test_iter, verbose=True)
		print("TEST accuracy: "+str(get_accuracy(confusion_matrix))+'\n')
		print(confusion_matrix)

		for k in responses:
			print(k, responses[k])

		# ----------  close session ------------

		# save final model to chekpoint file
		saver.save(sess, SAVE_NAME)

		coord.request_stop()
		coord.join(threads)



if __name__ == '__main__':

	run_model(	
		num_train_iterations=10000, 
		c3d_depth=0, 
		thresholding_approach="norm",
		training_dir="../../../train_ext/", 	
		training_dir_dataset_limit=0,
		validate_dir='', 
		testing_dir="../../../test_ext/",
		train_print_freq=10,
		validation_freq=1000,
		save_freq=1000,
		variable_update_freq=1000)

'''
	train_len = 10000

	train = True
	if(train):
		#01 03 05 07
		#09 11 13 15
		#17 19 21 23
		for i in range(11, 12, 2):
			#i=0
			print("Dataset Size: ", i, i*18)
			itr_exec = DQNExecutor(
							train_len=train_len, 
							training_dir=INP_FILE, 
							testing_dir=TEST_FILE,
							validate_dir=VAL_FILE, 
							#test_len=60, #20 from each label 
							c3d_depth=0,
							thresh_method="norm",
							dataset_limit=i)
			itr_exec.run_global_initalizers()
			itr_exec.finalize()
			itr_exec.run()
			tf.reset_default_graph()
'''