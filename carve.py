import sys
import math
import time
import numpy as np
from PIL import Image
from multiprocessing import Process, Pool
sys.setrecursionlimit(1500)  # height of the image must not exceed this

def distance(px1, px2):
	"""Euclidean distance between two pixels."""
	return sum([(a-b)**2 for a,b in zip(px1, px2)])


def get_distances(pixels):
	"""Return an array of distances to the closest neighbour for each pixel."""
	height, width, colour_depth = pixels.shape

	# we will store 3 copies of the bitmap (one for each horizontal offset)
	distances = [None, None, None]

	vertically_offset_pixels = np.roll(pixels, -1, axis=0)
	for horizontal_offset in (-1, 0, 1):
		pixels_offset = np.roll(vertically_offset_pixels, horizontal_offset, axis=1)
		# calculate colour distance -> sum((px1[c] - px2[c])**2 for c in colours)
		diff = (pixels - pixels_offset)**2
		colour_distance = np.sum(diff, axis=2)
		distances[horizontal_offset+1] = colour_distance

	# first column of the right-shifted bitmap shouldn't be used
	# or the last column of the left-shifted one
	# since they're just artifacts of np.roll-ing
	# TODO: is this really neccessary?
	# infty = 260101  # = 4*(255**2)+1 > max distance between two colours
	# distances[0][:, -1] = infty  # rolled to the left (delete last column)
	# distances[2][:, 0] = infty  # rolled to the right (delete first column)
	
	d = np.amin(distances, axis=0)

	# the last row should be all zeroes
	d[-1] = 0
	return d


def get_seam(distances):
	"""Return a list of column indices so that seam[row]=seam_column."""
	height, width = distances.shape
	if (distances.shape[0] == 1):
		return [int(np.argmin(distances))]
	
	last_row = distances[-1]
	second_last_row = distances[-2]

	# TODO: is it neccessary to disable the first and last element of the rolled arrays?
	distances[-2] = np.amin([
		second_last_row + np.roll(last_row,  0, axis=0),
		second_last_row + np.roll(last_row, +1, axis=0),
		second_last_row + np.roll(last_row, -1, axis=0),
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


def mark_seam(image, seam):
	"""Colour all pixels in a seam, given an array as returned by get_seam."""
	mark_colour = (255, 255, 255) if MODE == "RGB" else (255, 255, 255, 255)
	pixels = image.load()
	for row in range(len(seam)):
		pixels[seam[row], row] = mark_colour


def remove_seam(pixels, seam):
	"""Creates a copy of a bitmap without the seam."""
	height, old_width, _ = pixels.shape
	new_width = old_width - 1
	arr = np.zeros((height, new_width, 3))

	for row in range(height):
		arr[row] = np.concatenate((pixels[row][:seam[row]], pixels[row][seam[row]+1:]), axis=0)

	return Image.fromarray(np.uint8(arr))


def save_image(img, filename):
	try:
		img.save(filename)
	except OSError:
		time.sleep(0.1)
		save_image(img, filename)

if __name__ == "__main__":
	MODE = "RGB"

	if len(sys.argv) > 1:
		filename = sys.argv[1]
	else:
		filename = "broadway_tower.jpg"

	image = Image.open(filename).convert("RGB")

	iteration_count = 1000
	children = [] #  child processes for saving output images
	with Pool(processes=50) as pool:
		for i in range(iteration_count):
			print("Processing", (i+1), "/", iteration_count)

			arr = np.array(image)
			seam = get_seam(get_distances(arr))

			mark_seam(image, seam)
			seam_filename = "output/optimized/" + filename[:-4] + "_" + str(i).zfill(4) + "_seamed.png"
			pool.apply_async(save_image, (image, seam_filename))

			shrunk = remove_seam(arr, seam)

			out_filename = "output/optimized/" + filename[:-4] + "_" + str(i).zfill(4) + "_shrunk.png"
			pool.apply_async(save_image, (shrunk, out_filename))

			image = shrunk

		pool.close()
		print(f"Saving ...")
		pool.join()

	print("Done :)")
