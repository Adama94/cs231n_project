# Import outside libraries
import numpy as np
import matplotlib.pyplot as plt

# Import other files
from train_model import Config

def main():
	config = Config()
	output_images_filename = config.data_dir + "/predictions/val_predictions.npy"
	output_images = np.load(output_images_filename)
	
	for image in output_images:
		imgplot = plt.imshow(image)
		plt.show()

if __name__ == "__main__":
	main()
