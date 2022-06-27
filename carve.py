import sys
import threading
from multiprocessing import Pool

import cv2
import fire
import numpy as np
from PIL import Image
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk as skimage_disk
from tqdm import tqdm

MAX_IMAGE_HEIGHT = 1500

# height of the image must not exceed the recursion limit
sys.setrecursionlimit(MAX_IMAGE_HEIGHT)


def entropy_simple(pixels):
	"""Calculate the entropy for a given image in grayscale."""
	footprint = skimage_disk(3)
	r = pixels[:,:,0]*0.2989
	g = pixels[:,:,1]*0.5870
	b = pixels[:,:,2]*0.1140
	grayscale = np.array(r+g+b, dtype="uint8")
	ent = skimage_entropy(grayscale, footprint)
	scale = 255/np.max(ent)
	return ent*scale


def entropy_3ch(pixels):
	"""Calculate the entropy for a given image on 3 channels."""
	footprint = skimage_disk(3)
	r = pixels[:,:,0].astype('uint8')
	g = pixels[:,:,1].astype('uint8')
	b = pixels[:,:,2].astype('uint8')
	with Pool(processes=3) as pool:
		ret_r = pool.apply_async(skimage_entropy, (r, footprint))
		ret_g = pool.apply_async(skimage_entropy, (g, footprint))
		ret_b = pool.apply_async(skimage_entropy, (b, footprint))

		entropy_r = ret_r.get()
		entropy_g = ret_g.get()
		entropy_b = ret_b.get()
	ent = np.array(entropy_r + entropy_g + entropy_b)
	scale = 255/np.max(ent)
	return ent*scale


def saliency_spectral(pixels):
	"""Calculate the saliency map for a given image using OpenCV."""
	saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	(success, saliencyMap) = saliency.computeSaliency(pixels.astype('uint8'))
	if not success:
		raise ValueError("Cannot compute saliency.")
	return saliencyMap*255


def saliency_fine(pixels):
	"""Calculate the saliency map for a given image using OpenCV."""
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(pixels.astype('uint8'))
	if not success:
		raise ValueError("Cannot compute saliency.")
	return saliencyMap*255


def gradient_magnitude(pixels):
	"""Calculate the gradient magnitude for each pixel in an image.
	
	energy of (x, y) = sqrt(dx + dy)
	dx = sum(((x+1, y)[c] - (x-1, y)[r])**2 for c in [r, g, b])
	dy = sum(((x, y+1)[c] - (x, y-1)[r])**2 for c in [r, g, b])
	in other terms
	dx = sum((left-right)**2 for c in [r, g, b])
	dy = sum((up-down)**2 for c in [r, g, b])
	"""

	# shifted images
	up    = np.roll(pixels, -1, axis=0)
	down  = np.roll(pixels, +1, axis=0)
	left  = np.roll(pixels, -1, axis=1)
	right = np.roll(pixels, +1, axis=1)

	# handle edges
	up[-1] = up[-2].copy()
	down[0] = down[1].copy()
	left[:,-1] = left[:,-2].copy()
	right[:,0] = right[:,1].copy()

	dx = np.sum((left-right)**2, axis=2)
	dy = np.sum((up-down)**2, axis=2)

	return np.sqrt(dx + dy)


def saliency_plus_gradient(pixels):
	return saliency_spectral(pixels)/2 + gradient_magnitude(pixels)/2


def get_seam(distances):
	"""Return a list of column indices so that seam[row]=seam_column."""
	height, width = distances.shape
	if (distances.shape[0] == 1):
		return [int(np.argmin(distances))]
	
	last_row = distances[-1]
	second_last_row = distances[-2]

	last_row_C = np.roll(last_row,  0, axis=0)
	last_row_L = np.roll(last_row, +1, axis=0)
	last_row_R = np.roll(last_row, -1, axis=0)

	# disable the first and last element of the rolled arrays because they were wrapped
	infty = np.max(last_row + second_last_row)
	last_row_L[0] = infty
	last_row_R[-1] = infty

	distances[-2] = np.amin([
		second_last_row + last_row_C,
		second_last_row + last_row_L,
		second_last_row + last_row_R,
	], axis=0)

	try:
		seam = get_seam(distances[:-1])
	except RecursionError:
		print("Recursion depth exceeded. Image height must not exceed max recursion depth.")
		sys.exit(0)

	last_index = seam[-1]
	candidate_indices = [last_index, last_index-1, last_index+1]
	
	if -1 in candidate_indices:
		candidate_indices.remove(-1)
	if width in candidate_indices:
		candidate_indices.remove(width)

	candidate_values = [last_row[c] for c in candidate_indices]
	best_candidate_idx = np.argmin(candidate_values)

	next_index = candidate_indices[best_candidate_idx]
	seam.append(int(next_index))
	return seam


def mark_seam(image, seam, mark_colour=(255, 255, 255)):
	"""Colour all pixels in a seam, given an array as returned by get_seam."""
	pixels = image.load()
	for row in range(len(seam)):
		pixels[seam[row], row] = mark_colour


def remove_seam(pixels, seam):
	"""Create a copy of an image without the seam."""
	height, old_width, _ = pixels.shape
	new_width = old_width - 1
	arr = np.zeros((height, new_width, 3))

	for row in range(height):
		arr[row] = np.concatenate((pixels[row][:seam[row]], pixels[row][seam[row]+1:]), axis=0)

	return Image.fromarray(np.uint8(arr))

def save_seam(image, seam, filename):
	mark_seam(image, seam)
	save_image(image, filename)

def save_image(img, filename):
	img.save(filename)

def carve(
	in_filename,
	out_filename,
	*_,
	iteration_count=1000,
	save_seamed=True,
	save_shrunk=True,
	save_energy=True,
	energy_function=0
	):

	get_energy = energy_functions[energy_function]

	image = Image.open(in_filename).convert("RGB")

	threads = []
	for i in tqdm(range(iteration_count)):
		seam_filename = f"{out_filename}_{str(i).zfill(4)}_seamed.png"
		shrunk_filename = f"{out_filename}_{str(i).zfill(4)}_shrunk.png"
		energy_filename = f"{out_filename}_{str(i).zfill(4)}_energy.png"

		img_arr = np.array(image, dtype=np.longlong)
		energy = get_energy(img_arr)

		# save the energy map
		if save_energy:
			energy_image = Image.fromarray(energy).convert("RGB")
			x = threading.Thread(target=save_image, args=(energy_image, energy_filename))
			x.start()
			threads.append(x)

		seam = get_seam(energy)

		# mark seam and save
		if save_seamed:
			x = threading.Thread(target=save_seam, args=(image, seam, seam_filename))
			x.start()
			threads.append(x)

		shrunk = remove_seam(img_arr, seam)

		# remove seam and save
		if save_shrunk:
			x = threading.Thread(target=save_image, args=(shrunk, shrunk_filename))
			x.start()
			threads.append(x)

		image = shrunk
	
	# wait for everything to finish saving
	for thread in threads:
		thread.join()

	save_image(image, f"{out_filename}_final.png")

	print("Done :)")


def energy_demo(in_filename, out_filename, energy_function):
	carve(
		in_filename=in_filename,
		out_filename=out_filename,
		iteration_count=1,
		save_seamed=False,
		save_shrunk=False,
		save_energy=True,
		energy_function=energy_function
	)

energy_functions = [gradient_magnitude, saliency_spectral, saliency_fine, saliency_plus_gradient, entropy_3ch, entropy_simple]
if __name__ == "__main__":
	if len({"help", "--help", "-h"}.intersection(sys.argv)) != 0:
		print(f"Usage: {sys.argv[0]} [in_filename] [out_filename]")
		print("  in_filename is a path to the input image")
		print("  out_filename is a path to the output image, without the extension")
		print()
		print("Optional parameters:")
		for name, value in carve.__kwdefaults__.items():
			print(" --", name, "=", value, sep="")
		print()
		print(f"energy_function is the index of the energy function:")
		for i,f in enumerate(energy_functions):
			print("", i, f.__name__)
	else:
		fire.Fire(carve)
