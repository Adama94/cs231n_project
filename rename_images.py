import fnmatch
import os
from shutil import copyfile

count = 1
for file in os.listdir('.'):
	pattern = file[:-4]
	if pattern[1] == ".":
		if os.path.exists((pattern[2:] + ".jpg")):
			copyfile(file, "clean_data/image_" + str(count) + ".jpg")	
			copyfile(pattern[2:] + ".jpg", "clean_data/label_" + str(count) + ".jpg")
			count += 1	


