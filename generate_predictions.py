# Import outside libraries
import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt

# Import other files
from train_model import Config
from model import CancerDetectionSystem

# Setup logging level
logging.basicConfig(level=logging.INFO)

def initialize_model(session, model, train_dir):
	ckpt = tf.train.get_checkpoint_state(train_dir)
	v2_path = ckpt.model_checkpoint_path + ".index"	if ckpt else ""
	if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path)) or tf.gfile.Exists(v2_path):
		logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver = tf.train.Saver()
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		assert("No model parameters found. Exiting.")
	return model

def main(_):
	config = Config()
	
	val_images_filename = config.data_dir + "/val/val_images.npy"
	val_labels_filename = config.data_dir + "/val/val_labels.npy"

	val_images = np.load(val_images_filename)
	val_labels = np.load(val_labels_filename)

	with tf.Graph().as_default():
		model = CancerDetectionSystem(config)
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			
			# Load model	
			session.run(init)
			initialize_model(session, model, config.train_dir)
			
			# Save val images
			output_images = model.predict(session, val_images, val_labels)
			save_filename  = config.data_dir + "/predictions/val_predictions.npy"
			np.save(save_filename, output_images)

if __name__ == "__main__":
	tf.app.run()
