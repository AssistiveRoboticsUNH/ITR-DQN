import tensorflow as tf 

def weight_variable(name, shape):
	# creates a variable with given name and shape
	initial = tf.truncated_normal_initializer(stddev=0.1)
	return tf.get_variable(name, shape, initializer=initial)

def bias_variable(name, shape):
	# creates a variable with given name and shape
	initial = tf.constant(0.1, shape=shape)
	return tf.get_variable(name, initializer=initial)

def convolve_2d(input_tensor, W, b):
	# performs a 2d convolution and applies a leaky ReLu activation layer
	conv = tf.nn.conv2d(input_tensor, W, strides=[1,1,1,1], padding='VALID') + b
	return tf.nn.leaky_relu(conv, alpha=0.01)