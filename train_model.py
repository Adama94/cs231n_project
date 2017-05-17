# Import outside libraries
import tensorflow as tf
import numpy as np
from scipy import misc
import logging
import os
from os.path import join as pjoin
import time

# Import model/other functions
from model import CancerDetectionSystem

# Setup logging level and suppress tf warnings
logging.basicConfig(level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Config:
	lr = 0.0001
	opt = "adam"	

	data_dir = "../data"
	train_dir = "../train"
	log_dir = "log"
	
	image_shape = [720, 1128, 3]
	num_classes = 3
	n_epochs = 10
	batch_size = 8

def initialize_model(session, model, train_dir):
	ckpt = tf.train.get_checkpoint_state(train_dir)
	v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
	if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
		logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver = tf.train.Saver()
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		logging.info("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
		model.saver = tf.train.Saver()
		logging.info("Number of parameters: %d" % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
	return model

def initialize_dataset(data_dir, data_set):
	image_path = pjoin(data_dir, data_set)
	logging.info("Loading %s dataset..." % data_set)
	
	with open(image_path + "/info.txt", 'r') as f:
		num_images = int([x.rstrip() for x in f.readlines()][0])
		images = []
		labels = []

		for i in range(1, num_images + 1):
			image_name = image_path + "/image_" + str(i) + ".jpg"
			label_name = image_path + "/label_" + str(i) + ".jpg"
			
			image = misc.imread(image_name)
			label = misc.imread(label_name)
			
			images.append(image)
			labels.append(label)
		
		logging.info("Done...")
		return np.array(images), np.array(labels)
		

def initialize_train_dataset(data_dir):
	train_dataset = initialize_dataset(data_dir, "train")
	return train_dataset

def initialize_val_dataset(data_dir):
	val_dataset = initialize_dataset(data_dir, "val")
	return val_dataset

def initialize_test_dataset(data_dir):
	test_dataset = initialize_dataset(data_dir ,"test")
	return test_dataset

def preprocess_data(data):
	def is_green(pixel):
		r, g, b = pixel
		r, g, b = threshold(r, g, b)
		if g > 125:
			return True
		return False

	def is_red(pixel):	
		r, g, b = pixel
		r, g, b = threshold(r, g, b)
		if r > 125:
			return True
		return False
	
	def threshold(r, g, b):
		if r < 25:
			r = 0 
		if g < 25:
			g = 0
		if b < 25:
			b = 0
		return r, g, b

	images, labels = data
	new_labels = []
	for label in labels:
		new_label = np.zeros(label.shape[:2])
		for i in range(label.shape[0]):
			for j in range(label.shape[1]):
				if is_green(label[i][j]):
					new_label[i][j] = 0
				elif is_red(label[i][j]):
					new_label[i][j] = 1
				else:
					new_label[i][j] = 2
		new_labels.append(new_label)
	
	return images, np.array(new_labels)


def main(_):
	config = Config()
	
	# Set up logging infrastructure
	if not os.path.exists(config.log_dir):
		os.makedirs(config.log_dir)
	file_handler = logging.FileHandler(pjoin(config.log_dir, "log.txt"))
	logging.getLogger().addHandler(file_handler)
	
	# Load all datasets and save in numpy arrays once
	# train_data = initialize_train_dataset(config.data_dir)
	# val_data = initialize_val_dataset(config.data_dir)
	# test_data = initialize_test_dataset(config.data_dir)

	# train_images, train_labels = preprocess_data(train_data)
	# val_images, val_labels = preprocess_data(val_data)
	# test_images, test_labels = preprocess_data(test_data)
	
	train_images_filename = config.data_dir + "/train/train_images.npy"
	train_labels_filename = config.data_dir + "/train/train_labels.npy"
	val_images_filename = config.data_dir + "/val/val_images.npy"
	val_labels_filename = config.data_dir + "/val/val_labels.npy"
	test_images_filename = config.data_dir + "/test/test_images.npy"
	test_labels_filename = config.data_dir + "/test/test_labels.npy"
	
	# np.save(train_images_filename, train_images)
	# np.save(train_labels_filename, train_labels)
	# np.save(val_images_filename, val_images)
	# np.save(val_labels_filename, val_labels)
	# np.save(test_images_filename, test_images)
	# np.save(test_labels_filename, test_labels)
	# exit()

	# Load data from saved numpy files
	logging.info("Loading data from numpy binaries...")
	train_images = np.load(train_images_filename)
	train_labels = np.load(train_labels_filename)
	val_images = np.load(val_images_filename)
	val_labels = np.load(val_labels_filename)
	test_images = np.load(test_images_filename)
	test_labels = np.load(test_labels_filename)
	logging.info("Done...")

	train_data = (train_images, train_labels)
	val_data = (val_images, val_labels)
	test_data = (test_images, test_labels)
	
	with tf.Graph().as_default():
		logging.info("Building model...")
		start = time.time()
			
		# Initialize model
		model = CancerDetectionSystem(config)		
		logging.info("It took %.2f seconds", time.time() - start)

		init = tf.global_variables_initializer()
		with tf.Session() as session:
			
			# Initialize and train model
			session.run(init)			
			initialize_model(session, model, config.train_dir)
			model.train(session, train_data, val_data)

		# Output test accuracy
		# accuracy = model.test(session, test_dataset)
		# logging.info("Test accuracy: %.2f", accuracy)
		
if __name__ == "__main__":
	tf.app.run()
