import tensorflow as tf 

NUM_CLASSES = 101

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 256

def get_input_placeholder(batch_size):
	return tf.placeholder(tf.float32, 
			shape=(batch_size,
					NUM_FRAMES_PER_CLIP,
					CROP_SIZE,
					CROP_SIZE,
					CHANNELS),
			name="c3d_input_ph")

def get_output_placeholder(batch_size):
	return tf.placeholder(tf.int64, 
			shape=(batch_size),
			name="c3d_label_ph")

def get_variables():
	def _variable_on_cpu(name, shape, initializer):
		#with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
		return var

	def _variable_with_weight_decay(name, shape, stddev, wd):
		var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
		if wd is not None:
			weight_decay = tf.nn.l2_loss(var) * wd
			tf.add_to_collection('losses', weight_decay)
		return var


	with tf.variable_scope('var_name') as var_scope:
		weights = {
				'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
				'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
				'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
				'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
				'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
				'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
				'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
				'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
				'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
				'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
				'out': _variable_with_weight_decay('wout', [4096, NUM_CLASSES], 0.04, 0.005)
				}
		biases = {
				'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
				'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
				'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
				'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
				'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
				'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
				'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
				'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
				'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
				'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
				'out': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
				}
	return weights, biases

def conv3d(name, l_input, w, b):
		return tf.nn.bias_add(
			tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
			b)

def max_pool(name, l_input, k):
	return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def generate_feature_activations(input_ph, _weights, _biases, depth=4):
#def generate_feature_activations(input_ph, _weights, _biases):

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

	layers = [conv1, conv2, conv3, conv4, conv5]

	return layers[depth]