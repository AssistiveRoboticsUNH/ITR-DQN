import tensorflow as tf 
from basic_tf_commands import weight_variable, bias_variable, convolve_2d


# t = number of frames, h = height, w = width, c = number of channels
INPUT_DATA_SIZE = {"t": 1, "h":128, "w":512, "c":1}
OUT_SIZE = 60*252*32


def get_input_placeholder(batch_size):
	#Returns a placeholder for the Audio input.
	return tf.placeholder(tf.float32, 
			shape=(batch_size, INPUT_DATA_SIZE["h"], INPUT_DATA_SIZE["w"], INPUT_DATA_SIZE["c"]),
			name="aud_input_ph")

def get_output_placeholder(batch_size, num_label=2):
	# returns a placeholder for the Audio output (currently unused)
	return tf.placeholder(tf.int64,
			shape=(batch_size, 
				num_label),
			name="aud_label_ph")


def get_variables(model_name="aud"):
	#Define all of the variables for the convolutional layers of the Audio model

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

	#perform 2x2 pooling to reduce the activation map size
	return tf.layers.max_pooling2d(conv_tensor, pool_size=[2, 2], strides=2)