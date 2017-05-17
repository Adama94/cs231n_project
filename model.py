# Import outside libraries
import tensorflow as tf
import numpy as np
import time
import logging
import os
from os.path import join as pjoin

# Setup logging level
logging.basicConfig(level=logging.INFO)

class CancerDetectionSystem(object):
	def __init__(self, config):
		
		self.config = config
		
		self.batch_shape = [None] + config.image_shape
		self.labels_shape = self.batch_shape[:3]
		self.batch_shape = tuple(self.batch_shape)
		self.labels_shape = tuple(self.labels_shape)
		
		self.images_placeholder = tf.placeholder(tf.float32, shape=self.batch_shape)
		self.labels_placeholder = tf.placeholder(tf.int32, shape=self.labels_shape)

		self.final_images = self.setup_system()	
		self.loss = self.setup_loss(self.final_images, self.labels_placeholder)
		self.train_op = self.setup_training_op(self.loss)
		
		self.saver = tf.train.Saver()
		
	def setup_system(self):
		images = self.images_placeholder
		config = self.config
		

		init = tf.contrib.layers.xavier_initializer()		
		output_channels_1 = 64
		filter_1 = tf.get_variable("filter_1", shape=(3, 3, 3, output_channels_1), \
					 initializer=init)
		layer_1 = tf.nn.conv2d(images, filter_1, [1, 1, 1, 1], "SAME")
		h_1 = tf.nn.relu(layer_1)
		output_channels_2 = 32
		filter_2 = tf.get_variable("filter_2", shape=(3, 3, output_channels_1, output_channels_2), \
					initializer=init)
		layer_2 = tf.nn.conv2d(h_1, filter_2, [1, 1, 1, 1], "SAME")
		h_2 = tf.nn.relu(layer_2)
		output_channels_3 = 3
		filter_3 = tf.get_variable("filter_3", shape=(3, 3, output_channels_2, output_channels_3), \
					initializer=init)
		scores = tf.nn.conv2d(h_2, filter_3, [1, 1, 1, 1], "SAME")

		return scores
		

	def setup_loss(self, prediction, truth):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=truth, logits=prediction)
		average = tf.reduce_mean(loss, axis=[1, 2])
		return tf.reduce_mean(average)

	def get_optimizer(self, opt):
		if opt == "adam":
			optfn = tf.train.AdamOptimizer
		elif opt == "sgd":
			optfn = tf.train.GradientDescentOptimizer
		else:
			assert (False)
		return optfn

	def setup_training_op(self, loss):
		lr = self.config.lr
		opt = self.config.opt
		train_op = self.get_optimizer(opt)(learning_rate=lr).minimize(loss)
		return train_op

	def train_on_batch(self, session, images, labels):
		input_feed = {
			self.images_placeholder: images,
			self.labels_placeholder: labels
		}
		
		output_feed = [self.train_op, self.loss]
		_, loss = session.run(output_feed, input_feed)
		print(loss)
		return loss
	
	def minibatches(self, data, batch_size):
		images, labels = data
		data_size = len(images)
		indices = np.arange(data_size)
		np.random.shuffle(indices)
		
		index_partitions = [indices[x:x+batch_size] for x in range(0, data_size, batch_size)]
		minibatches = []
		for index_partition in index_partitions:
			partition_images = [images[x] for x in index_partition]
			partition_labels = [labels[x] for x in index_partition]
			minibatches.append((partition_images, partition_labels))
		return minibatches

	def validate(self, session, val_data):
		images, labels = val_data
		
		input_feed = {
			self.images_placeholder: images,
			self.labels_placeholder: labels
		}

		output_feed = [self.loss]
		outputs = session.run(output_feed, input_feed)
		return outputs[0]
	
	def predict(self, session, image, label):
		input_feed = {
			self.images_placeholder: image,
			self.labels_placeholder: label
		}
		output_feed = [self.final_images]
		outputs = session.run(output_feed, input_feed)
		logits = outputs[0]
		
		final_images = []
		for logit_map in logits:
			image = np.zeros(logit_map.shape)
			indices = np.argmax(logit_map, axis=2)
			image[indices == 0] = [0.0, 1.0, 0.0]
			image[indices == 1] = [1.0, 0.0, 0.0]
			final_images.append(image)
		return np.array(final_images)

	def run_epoch(self, session, train_data, val_data):
		
		train_loss = 0.0
		minibatches = self.minibatches(train_data, self.config.batch_size)
		num_minibatches = len(minibatches)

		for i, batch in enumerate(minibatches):
			images, labels = batch
			batch_loss = self.train_on_batch(session, images, labels)
			train_loss += batch_loss

		train_loss /= num_minibatches

		val_loss = self.validate(session, val_data)
		
		return train_loss, val_loss
	
	def train(self, session, train_data, val_data):
		
		train_images, train_labels = train_data
		val_images, val_labels = val_data
		
		tic = time.time()
		params = tf.trainable_variables()
		num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
		toc = time.time()
		logging.info("Number of params: %d (retrieval took %f seconds)" % (num_params, toc - tic))
		
		best_val_loss = None
		train_dir = self.config.train_dir
		for epoch in range(self.config.n_epochs):
			logging.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
			train_loss, val_loss = self.run_epoch(session, train_data, val_data)
			logging.info("Traininng Loss: %f", train_loss)
			logging.info("Validation Loss: %f", val_loss)
			
			if best_val_loss == None or val_loss < best_val_loss:
				best_val_loss = val_loss
				if self.saver:
					logging.info("New best loss! Saving model in %s", train_dir)
					save_path = pjoin(train_dir, "train")
					self.saver.save(session, save_path)
					
		return best_val_loss









