import os
import subprocess
import numpy as np
import binvox_rw
import mayavi.mlab


path_to_dataset = "" # path to your raw_meshes folder
render = False # turn to true for rendering



master_array = []
def generatevoxels(stlfile):
	os.system("binvox -cb -pb -d 32 " + str(stlfile))
	try:
		binvoxpath = stlfile.split(".")[0] + ".binvox"
		with open(binvoxpath, 'rb') as f:
			model = binvox_rw.read_as_3d_array(f)
		
		voxel_array = model.data
		voxel_array = voxel_array * 1 # convert to zero and one
		print voxel_array
		print voxel_array.shape

		#model[False] = 0
		#model[True]  = 1
		
		#print voxel_array
		#print type(voxel_array)
		#print voxel_array.shape'
		if render == True:
			render_in_3d(voxel_array)
		
		voxel_array = np.expand_dims(voxel_array, -1)
		
		os.system("rm " + str(binvoxpath))

		master_array.append(voxel_array)
		print "Here's how many have been processed: " + str(len(master_array))
		if len(master_array) % 500 == 0:
			outfile = open("data/thingi10k_" + str(len(master_array)) + ".npy", "w")
			np.save(outfile, np.array(master_array))
	except Exception as e:
		print e
		
def render_in_3d(array):
	xx, yy, zz = np.where(array > 0.95)
	mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=(0.96, 0.96, 0.96), scale_factor=1)
	mayavi.mlab.show()

if path_to_dataset[-1] != "/":
	path_to_dataset
for i in os.listdir(path_to_dataset):
	generatevoxels(path_to_dataset + str(i))
