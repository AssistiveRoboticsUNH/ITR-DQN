import tensorflow as tf 



# t = number of frames, h = height, w = width, c = number of channels
INPUT_DATA_SIZE = {"t": 256, "h":112, "w":112, "c":3}

def get_input_placeholder(batch_size):
	# returns a placeholder for the C3D input
	return tf.placeholder(tf.float32, 
			shape=(batch_size, INPUT_DATA_SIZE["t"], INPUT_DATA_SIZE["h"], INPUT_DATA_SIZE["w"], INPUT_DATA_SIZE["c"]),
			name="c3d_input_ph")


def get_output_placeholder(batch_size):
	# returns a placeholder for the C3D output (currently unused)
	return tf.placeholder(tf.float32, 
			shape=(batch_size),
			name="c3d_label_ph")


def get_variables():
	'''Define all of the variables for the convolutional layers of the C3D model. 
	We ommit the FC layers as these layers are used to perform reasoning and do 
	not contain feature information '''

	def weight_variable(name, shape):
		# creates a variable (weight or bias) with given name and shape
		initial = tf.truncated_normal_initializer(stddev=0.04)
		return tf.get_variable(name, shape, initializer=initial)

	with tf.variable_scope('var_name') as var_scope:
		weights = {
				'wc1': weight_variable('wc1', [3, 3, 3, 3, 64]),
				'wc2': weight_variable('wc2', [3, 3, 3, 64, 128]),
				'wc3a': weight_variable('wc3a', [3, 3, 3, 128, 256]),
				'wc3b': weight_variable('wc3b', [3, 3, 3, 256, 256]),
				'wc4a': weight_variable('wc4a', [3, 3, 3, 256, 512]),
				'wc4b': weight_variable('wc4b', [3, 3, 3, 512, 512]),
				'wc5a': weight_variable('wc5a', [3, 3, 3, 512, 512]),
				'wc5b': weight_variable('wc5b', [3, 3, 3, 512, 512])
				}
		biases = {
				'bc1': weight_variable('bc1', [64]),
				'bc2': weight_variable('bc2', [128]),
				'bc3a': weight_variable('bc3a', [256]),
				'bc3b': weight_variable('bc3b', [256]),
				'bc4a': weight_variable('bc4a', [512]),
				'bc4b': weight_variable('bc4b', [512]),
				'bc5a': weight_variable('bc5a', [512]),
				'bc5b': weight_variable('bc5b', [512])
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

	def conv3d(name, l_input, w, b):
		# performs a 3d convolution
		return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),b)

	def max_pool(name, l_input, k):
		# performs a 2x2 max pool operation in 3 dimensions
		return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

	# Convolution Layer
	conv1 = conv3d('conv1', input_ph, _weights['wc1'], _biases['bc1'])
	conv1 = tf.nn.relu(conv1, 'relu1')
	pool1 = max_pool('pool1', conv1, k=1)

	# Convolution Layer
	conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
	conv2 = tf.nn.relu(conv2, 'relu2')
	pool2 = max_pool('pool2', conv2, k=2)

	# Convolution Layer
	conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
	conv3 = tf.nn.relu(conv3, 'relu3a')
	conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
	conv3 = tf.nn.relu(conv3, 'relu3b')
	pool3 = max_pool('pool3', conv3, k=2)

	# Convolution Layer
	conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
	conv4 = tf.nn.relu(conv4, 'relu4a')
	conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
	conv4 = tf.nn.relu(conv4, 'relu4b')
	pool4 = max_pool('pool4', conv4, k=2)

	# Convolution Layer
	conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
	conv5 = tf.nn.relu(conv5, 'relu5a')
	conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
	conv5 = tf.nn.relu(conv5, 'relu5b')

	# an array of convolution layers to select from
	layers = [conv1, conv2, conv3, conv4, conv5]

	return layers[depth]