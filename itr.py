import tensorflow as tf 
from basic_tf_commands import weight_variable, bias_variable, convolve_2d





NUM_FEATURES = [64, 128, 256, 512, 512]
LENGTH_OF_FRAME = [256,256, 128,64,32]

def calculate_output_size(c3d_depth, itr_depth=4):
	dimensions = [NUM_FEATURES[c3d_depth], LENGTH_OF_FRAME[c3d_depth]]
	for i in range(itr_depth):
		for d in range(len(dimensions)):
			dimensions[d] = (dimensions[d]-3) + 1

	return dimensions[0]*dimensions[1]*32

def get_input_placeholder(batch_size, c3d_map_depth):
	'''
	Returns a placeholder for the ITR input. Because the size of the input is dependent
	on the size of the C3D activation layer chosen we require it as a parameter
		-c3d_map_depth: the depth at which the C3D activation map is beign generated
	'''
	return tf.placeholder(tf.float32, 
			shape=(batch_size, 
				NUM_FEATURES[c3d_map_depth], 
				LENGTH_OF_FRAME[c3d_map_depth], 
				1),
			name="itr_input_ph")

def get_output_placeholder(batch_size, num_label=2):
	# returns a placeholder for the C3D output (currently unused)
	return tf.placeholder(tf.int64,
			shape=(batch_size, 
				num_label),
			name="itr_label_ph")


def get_variables(model_name="itr"):
	#Define all of the variables for the convolutional layers of the ITR model

	with tf.variable_scope(model_name) as var_scope:
		weights = {
				'W_0': weight_variable('W_0', [3, 3, 1, 16]),
				'W_1': weight_variable('W_1', [3, 3, 16, 16]),
				'W_2': weight_variable('W_2', [3, 3, 16, 32]),
				'W_3': weight_variable('W_3', [3, 3, 32, 32])
				}
		biases = {
				'b_0': bias_variable('b_0', [16]),
				'b_1': bias_variable('b_1', [16]),
				'b_2': bias_variable('b_2', [32]),
				'b_3': bias_variable('b_3', [32])
				}
	return weights, biases


def generate_activation_map(input_ph, _weights, _biases, depth=4):
	'''Generates the activation map for a given input from a specific depth
				-input_ph: the input placeholder, should have been defined using the 
					"get_input_placeholder" function
				-_weights: weights used to convolve the input, defined in the 
					"get_variables" function
				-_biases: biases used to convolve the input, defined in the 
					"get_variables" function
				-depth: the depth at which the activation map should be extracted (an 
					int between 0 and 4)
	'''
	conv_tensor = input_ph
	for l in range(depth):
		conv_tensor = convolve_2d(conv_tensor, _weights["W_"+str(l)], _biases["b_"+str(l)])
	return conv_tensor