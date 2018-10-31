import numpy as np
import tensorflow as tf 

import c3d, itr, aud
from basic_tf_commands import weight_variable, bias_variable

NUM_LABEL = 3

# define placeholders
def get_placeholders(batch_size=1, c3d_depth=0):
	'''
	Define the placeholders used by the network. 
		-c3d_depth: the depth at which the c3d activation map should be drawn
	'''

	return {
		# imported placeholders
		"c3d_in": c3d.get_input_placeholder(batch_size),
		"itr_in": itr.get_input_placeholder(batch_size, c3d_depth),

		"aud_in": aud.get_input_placeholder(batch_size),

		# system specific placeholders
		"pt_in": tf.placeholder(tf.float32,shape=(batch_size, 1),name="pt_ph"),
		"system_out": tf.placeholder(tf.float32,shape=(batch_size, NUM_LABEL),name="out_ph")
		}


def get_variables(c3d_depth=0):
	'''
	Define the variables used by the network. 
		-c3d_depth: the depth at which the c3d activation map should be drawn
	'''

	# imported variables
	itr_w, itr_b = itr.get_variables()

	aud_w, aud_b = aud.get_variables()

	# system specific variables
	system_w = {"W_0": weight_variable("W_0", [itr.calculate_output_size(c3d_depth)+aud.OUT_SIZE, 1]),
							"W_1": weight_variable("W_1", [2, NUM_LABEL])}
	system_b = {"b_0": bias_variable("b_0", [1]),
							"b_1": bias_variable("b_1", [NUM_LABEL])}

	#coalated variables
	weights = {
						"itr": itr_w,
						"aud": aud_w,
						"system": system_w}
	biases = {
						"itr": itr_b,
						"aud": aud_b,
						"system": system_b}

	return weights, biases

def list_variables(weights, biases):
	return {
		"itr": list( set(weights["itr"].values() + biases["itr"].values()) ),
		"aud": list( set(weights["aud"].values() + biases["aud"].values()) ),
		"system": list( set(weights["system"].values() + biases["system"].values()) )
		}

def get_predicted_values(placeholders, _weights, _biases, c3d_depth=0):
	'''
	Define the full structure of the network. 
		-placeholders: a set of placeholder variables established by running the
			"get_placeholders" function
		-weights: a set of weights established by running the
			"get_variables" function
		-weights: a set of biases established by running the
			"get_variables" function
	'''

	# Generate the output of the ITR network
	itr_activation_map = itr.generate_activation_map(placeholders["itr_in"], _weights["itr"], _biases["itr"])
	itr_activation_map = tf.reshape(itr_activation_map, (1, itr.calculate_output_size(c3d_depth)))

	# Generate the output of the Audio network
	aud_activation_map = aud.generate_activation_map(placeholders["aud_in"], _weights["aud"], _biases["aud"])
	aud_activation_map = tf.reshape(aud_activation_map, (1, aud.OUT_SIZE))

	# Combine the ITR and Audio networks into a single vector
	combined_vector = tf.concat((itr_activation_map, aud_activation_map), 1)
	
	# Generate pre_q values using a fully connected layer
	pre_q = tf.matmul(combined_vector, _weights["system"]["W_0"], name="pre_q_matmul") + _biases["system"]["b_0"]

	# Combine the pre_q values with pt (the number of prompts) and compute final q-values
	pre_q = tf.concat((pre_q, placeholders["pt_in"]), 1)
	return tf.matmul(pre_q, _weights["system"]["W_1"], name="q_matmul") + _biases["system"]["b_1"]

def classifier(model):
	'''
	Outputs the best available action
		-model: a TF model as would be generated via the "get_predicted_values" 
			function
	'''
	return tf.argmax(model,1)

def optimizer(placeholders, model, alpha=1e-3):
	'''
	Optimizes the network
		-placeholders: a set of placeholder variables established by running the
			"get_placeholders" function
		-model: a TF model as would be generated via the "get_predicted_values" 
			function
		-alpha: the learning rate for the Adam optimizer
	'''
	#loss = tf.clip_by_value(placeholders["system_out"],1e-10,1) - tf.clip_by_value(model,1e-10,100)
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=placeholders["system_out"], logits=model)
	return tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)




