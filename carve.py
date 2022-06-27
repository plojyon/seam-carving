from asyncio import _unregister_task
import cv2
import sys
import math
import time
import numpy as np
from PIL import Image, ImageCms
from multiprocessing import Process, Pool

MAX_IMAGE_HEIGHT = 1500

# height of the image must not exceed the recursion limit
sys.setrecursionlimit(MAX_IMAGE_HEIGHT)


def get_energy(pixels):
	"""Take an array of pixel data and return an array of energies."""

	# energy of (x, y) = sqrt(dx + dy)
	# dx = sum(((x+1, y)[c] - (x-1, y)[r])**2 for c in [r, g, b])
	# dy = sum(((x, y+1)[c] - (x, y-1)[r])**2 for c in [r, g, b])
	# in other terms
	# dx = sum((left-right)**2 for c in [r, g, b])

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
	in_filename="broadway_tower.jpg",
	out_filename="output/optimized/broadway_tower",
	iteration_count=800,
	save_seamed=True,
	save_shrunk=True,
	):

	image = Image.open(in_filename).convert("RGB")

	with Pool(processes=50) as pool:
		for i in range(iteration_count):
			print(f"Processing {i+1}/{iteration_count}")
			seam_filename = f"{out_filename}_{str(i).zfill(4)}_seamed.png"
			shrunk_filename = f"{out_filename}_{str(i).zfill(4)}_shrunk.png"

			img_arr = np.array(image, dtype=np.longlong)
			seam = get_seam(get_energy(img_arr))
			
			# mark seam and save
			if save_seamed:
				pool.apply_async(save_seam, (image, seam, seam_filename))
			
			# remove seam and save
			if save_shrunk:
				shrunk = remove_seam(img_arr, seam)
				pool.apply_async(save_image, (shrunk, shrunk_filename))

			image = shrunk

		pool.close()
		print("Saving ...")
		pool.join()

	print("Done :)")

if __name__ == "__main__":
	carve()