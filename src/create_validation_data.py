import os
import shutil 
from random import randrange
    
source_parent = '../data/train/' 
target_parent = '../data/validation/test/'
    
def move_files(source_dir, target_dir):
	file_names = os.listdir(source_dir)
	for file_name in file_names:
		if randrange(5) == 1:
	    		shutil.move(os.path.join(source_dir, file_name), target_dir)

def copy_files(source_dir, target_dir):
	file_names = os.listdir(source_dir)
	for file_name in file_names: 
    		shutil.copy(os.path.join(source_dir, file_name), target_dir)
	    		
move_files(source_parent + 'buy', target_parent + 'buy')
move_files(source_parent + 'sell', target_parent + 'sell') 
