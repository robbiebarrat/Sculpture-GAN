# Sculpture-GAN
3D DCGAN inspired GAN trained on 32x32x32 voxelizations of [Thingi10k](https://ten-thousand-models.appspot.com/) - a corpus of 10,000 3D printable objects and sculptures - as a result, the generated sculptures are almost always 3D-printable.

# Usage

There are really only 3 files you need to use to generate your own 3D shapes.

## generate_dataset.py

You need to download [Thingi10k](https://ten-thousand-models.appspot.com/), and in line 8 of generate_dataset.py, set the variable `path_to_dataset` to be the path to your `raw_meshes` folder from your thingi10k download.

Next, just run the file with `python generate_dataset.py`, and it will start generating numpy arrays that contain all of the  now-voxelized models in thingi10k and save them to your `data/` folder.

## train.py

On line 23 of `train.py`, point it to the desired (usually the largest) .npy file in your `data/` folder by setting `numpy_array_saved` equal to its path.

Then, you can just run `python train.py` and trained network checkpoints will start to populate your `network_checkpoints/` folder.

## visualize.py

Point `visualize.py` to a generator checkpoint, and it will allow you to see the models it can generate, with the option to save as PNG, obj, etc.
