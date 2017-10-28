# turns STL into plottable 3d (64x64x64) numpy array
from pprint import pprint
import numpy as np
import os
import mayavi.mlab

array_to_visualize = "" # path to the array the network generated

def render_in_3d(array):
	xx, yy, zz = np.where(array > 0.8)

	mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=(0.96, 0.96, 0.96), scale_factor=1)

	mayavi.mlab.show()

for i in range(100):
	array = np.squeeze(np.load(array_to_visualize)[i])

	render_in_3d(array)
