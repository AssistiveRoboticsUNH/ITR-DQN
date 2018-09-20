from __future__ import print_function

import tensorflow as tf 

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 256

NUM_EVENT = [64, 128, 256, 512, 512]
NUM_MOMENT = [256,256, 128,64,32]#156
NUM_LABEL = 2

img_dimension = 128
opt_dimension = 64

IMG_DTYPE = {
	"name": "img",
	"img_h": 480, 
	"img_w": 640, 
	"num_c": 3,
	"dimension": 3,
	"cmp_h": img_dimension,
	"cmp_w": img_dimension
}

OPT_DTYPE = {
	"name": "opt",
	"img_h": opt_dimension, 
	"img_w": opt_dimension,
	"num_c": 1,
	"dimension": 3,
	"cmp_h": opt_dimension,
	"cmp_w": opt_dimension
}

AUD_DTYPE = {	
	"name": "aud",
	"img_h": 128, 
	"img_w": 2,
	"num_c": 1,
	"dimension": 2,
	"cmp_h": 128,
	"cmp_w": 512
}

IC_DTYPE = [IMG_DTYPE, AUD_DTYPE]

EVENT_DTYPE = {	
	"name": "event",
	"num_c": 1,
	"dimension": 2,
	"cmp_h": NUM_EVENT,
	"cmp_w": NUM_MOMENT
}


EVENT_FILTER = {
	"Dimension":2,
	"Depth":4,
	"FilterShape":[3,3],
	"Stride":[1,1],
	"Padding":[0,0]
}
'''
EVENT_FILTER = {
	"Dimension":2,
	"Depth":3,
	"FilterShape":[[3,3], [5,5], [5,5]],
	"Stride":[[1,1], [3,3], [3,3]],
	"Padding":[[1,1], [2,2], [2,0]]
}
'''

MODAL_INPUTS = 3

NUM_FILTERS = [16,16,32,32,NUM_LABEL]#[16,32,128,NUM_LABEL]#
MAX_DEPTH = len(NUM_FILTERS)-1 # the length of the CNN hidden layers

def get_input_placeholder(batch_size, map_depth):
	dtype = EVENT_DTYPE
	#data_frame_size = dtype["cmp_h"][map_depth] * dtype["cmp_w"] * dtype["num_c"]

	return tf.placeholder(tf.float32, 
			shape=(batch_size,
				dtype["cmp_h"][map_depth], 
				dtype["cmp_w"][map_depth], 
				dtype["num_c"]),
			name="cnn2d_input_ph")

def get_output_placeholder(batch_size):
	return tf.placeholder(tf.int64,
			shape=(batch_size, NUM_LABEL),
			name="cnn2d_label_ph")

def _layer_output_size(layer, map_depth):
	# gets the length of a flattened output for the given layer
	size = 0
	f = EVENT_FILTER

	#define input shape
	dimensionSize = [EVENT_DTYPE["cmp_h"][map_depth], EVENT_DTYPE["cmp_w"][map_depth]]

	#generate output size for layer
	for i in range(layer):
		if(f["Depth"] > i):
			for d in range(len(dimensionSize)):
				val = (dimensionSize[d] - f["FilterShape"][d] + 
						2*f["Padding"][d])/float(f["Stride"][d]) + 1
				'''
				val = (dimensionSize[d] - f["FilterShape"][i][d] + 
						2*f["Padding"][i][d])/float(f["Stride"][i][d]) + 1
				'''
				assert(float(int(val)) == val, "output cannot be converted to integer")
				dimensionSize[d] = int(val)

		print("dimensionSize: ", dimensionSize)
	#dimensionSize.append(NUM_FILTERS[layer-1])
	print("dimensionSize_final: ", dimensionSize)
	#sum dimensions
	prod = 1
	for d in range(len(dimensionSize)):
		prod *= dimensionSize[d]
	size += prod

	print("size: ", size)

	return size

def get_variables(map_depth, name="cnn2d"):
	# get appropriate filter shape for CNN layer

	# variable shapes
	#-----------------------------

	def weight_shape_2d(layer):
		#filter_shape = EVENT_FILTER["FilterShape"][layer][:]
		filter_shape = EVENT_FILTER["FilterShape"][:]

		if(layer == 0):
			num_filter_inputs = EVENT_DTYPE["num_c"]
		else:
			num_filter_inputs = NUM_FILTERS[layer-1]
		filter_shape.append(num_filter_inputs)

		num_filter_outputs = NUM_FILTERS[layer]
		filter_shape.append(num_filter_outputs)
		
		return filter_shape

	def weight_shape_fc_end(layer, input_size=1):
		# returns the shape for a Fully Connected layer located at the end of the model
		assert layer > 0
		return [NUM_FILTERS[layer-1]*input_size, NUM_FILTERS[MAX_DEPTH]]

	def bias_shape(layer):
		# returns the shape for a bias layer
		return [NUM_FILTERS[layer]]

	def bias_shape_end():
		# returns the shape for a bias layer located at the end of the model
		return bias_shape(MAX_DEPTH)

	# variable initializers
	#-----------------------------

	# generate weight variables
	def weight_variable(name, shape):
		initial = tf.truncated_normal_initializer(stddev=0.1)
		return tf.get_variable(name, shape, initializer=initial)
		
	# generate bias variables
	def bias_variable(name, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.get_variable(name, initializer=initial)

	# make variables
	#---------------------------------

	# Generate the CNN variables
	weights, biases = {},{}

	with tf.variable_scope(name) as scope:
		for l in range(MAX_DEPTH):
			W_key, b_key = "W_"+str(l), "b_"+str(l)
				
			weights[W_key] = weight_variable(W_key, weight_shape_2d(l))
			biases[b_key] = bias_variable(b_key, bias_shape(l))

		# Generate the FC variables
		W_key, b_key = "W_fc", "b_fc"

		weights[W_key] = weight_variable(W_key, 
			weight_shape_fc_end(MAX_DEPTH, _layer_output_size(MAX_DEPTH, map_depth)))
		biases[b_key] = bias_variable(b_key, bias_shape_end())

	return weights, biases

def generate_classification(input_ph, _weights, _biases, map_depth, depth=MAX_DEPTH):
	print("generate_classification: ", depth)
	activations = generate_feature_activations(input_ph, _weights, _biases, depth)

	W, b = "W_fc", "b_fc"

	activations = tf.reshape(activations, [-1, _layer_output_size(depth, map_depth) * NUM_FILTERS[depth-1]])

	return tf.matmul(activations, _weights[W]) + _biases[b]

#obtain the activations for a particular depth of the network
def generate_feature_activations(input_ph, _weights, _biases, depth=MAX_DEPTH, filter_type=EVENT_FILTER, model_name="cnn2d"):
	#-------------------
	# Helper functions
	#-------------------

	def convolve(input_tensor, dimensions, W, b, stride, padding):
		# pad input in 3 Dimensions
		shape = [[0,0]]
		stride_size = [1]
		for i in range(dimensions):
			shape.append([padding[i], padding[i]])
			stride_size.append(stride[i])
		shape.append([0, 0])
		stride_size.append(1)

		padded_tensor = tf.pad(input_tensor, shape, "CONSTANT")

		# perform 2D Convolution
		conv = tf.nn.conv2d(padded_tensor, W, 
				strides=stride_size, padding='VALID') + b
		
		return tf.nn.relu(conv)

	#-------------------
	# 2D CNN 
	#-------------------



	with tf.variable_scope(model_name) as scope:

		conv_tensor = input_ph

		for l in range(depth):

			print("conv_shape", conv_tensor.get_shape())

			W, b = "W_"+str(l), "b_"+str(l)
			
			conv_tensor = convolve(
					conv_tensor, 
					filter_type["Dimension"], 
					_weights[W], 
					_biases[b], 
					filter_type["Stride"], 
					filter_type["Padding"]
					#filter_type["Stride"][l], 
					#filter_type["Padding"][l]
					)

		return conv_tensor