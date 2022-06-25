import sys
import math
import time
import numpy as np
from PIL import Image
from multiprocessing import Process, Pool

MAX_IMAGE_HEIGHT = 1500

sys.setrecursionlimit(MAX_IMAGE_HEIGHT)  # height of the image must not exceed this
# 4*(255**2) = max distance between two colours
infty = (4*(255**2)+1)*MAX_IMAGE_HEIGHT  # > maximum possible seam cost
if infty*MAX_IMAGE_HEIGHT > np.iinfo(np.longlong).max // 2:
	print("Max image height is too big")
	sys.exit(0)  # might work anyway, delete this line if you want


def distance(px1, px2):
	"""Euclidean distance between two pixels."""
	return sum([(a-b)**2 for a,b in zip(px1, px2)])


def get_distances(pixels):
	"""Return an array of distances to the closest neighbour for each pixel."""
	height, width, colour_depth = pixels.shape

	# we will store 3 copies of the bitmap (one for each horizontal offset)
	distances = []

	vertically_offset_pixels = np.roll(pixels, -1, axis=0)
	for horizontal_offset in (-1, 0, 1):
		pixels_offset = np.roll(vertically_offset_pixels, horizontal_offset, axis=1)
		# calculate colour distance -> sum((px1[c] - px2[c])**2 for c in colours)
		diff = (pixels - pixels_offset)**2
		colour_distance = np.sum(diff, axis=2)
		distances.append(colour_distance)

	# first column of the right-shifted bitmap shouldn't be used
	# or the last column of the left-shifted one
	# since they're just artifacts of np.roll-ing
	# TODO: is this really neccessary?
	distances[0][:, -1] = infty  # rolled to the left (delete last column)
	distances[2][:, 0] = infty  # rolled to the right (delete first column)

	d = np.amin(distances, axis=0)

	#* Remove in production for better speed
	if (d >= infty).any() or (d < 0).any():
		raise ValueError("Sanity check failed.")

	# the last row should be all zeroes
	d[-1] = 0
	return d


def get_seam(distances):
	"""Return a list of column indices so that seam[row]=seam_column."""
	height, width = distances.shape
	if (distances.shape[0] == 1):
		print("distances:", distances)
		print("Expected cost:", np.min(distances)) #! DEBUG
		return [int(np.argmin(distances))]

	last_row = distances[-1]
	second_last_row = distances[-2]

	# TODO: is it neccessary to disable the first and last element of the rolled arrays?
	last_row_C = np.roll(last_row,  0, axis=0)
	last_row_L = np.roll(last_row, +1, axis=0)
	last_row_R = np.roll(last_row, -1, axis=0)
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

	# remove edge cases
	if -1 in candidate_indices:
		candidate_indices.remove(-1)
	if width in candidate_indices:
		candidate_indices.remove(width)

	candidate_values = [last_row[c] for c in candidate_indices]
	best_candidate_idx = np.argmin(candidate_values)

	next_index = candidate_indices[best_candidate_idx]
	seam.append(next_index)
	
	return seam


def mark_seam(image, seam, arr): #! DEBUG: arr
	"""Colour all pixels in a seam, given an array as returned by get_seam."""
	mark_colour = (255, 255, 255) if image.mode == "RGB" else (255, 255, 255, 255)
	np.seterr(all='raise') #! DEBUG
	pixels = image.load()
	sum = 0 #! DEBUG
	sum2 = 0 #! DEBUG
	org_arr = np.array(image, dtype=np.longlong)
	prev = org_arr[0, seam[0]] #! DEBUG
	for row in range(len(seam)):
		sum2 += distance(prev, org_arr[row, seam[row]]) #! DEBUG
		prev = org_arr[row, seam[row]] #! DEBUG
		
		sum += arr[row, seam[row]] #! DEBUG
		pixels[seam[row], row] = mark_colour
	print("Actual seam cost:", sum) #! DEBUG
	print("Actual distance sum:", sum2) #! DEBUG


def remove_seam(pixels, seam):
	"""Creates a copy of a bitmap without the seam."""
	height, old_width, _ = pixels.shape
	new_width = old_width - 1
	arr = np.zeros((height, new_width, 3))

	for row in range(height):
		arr[row] = np.concatenate((pixels[row][:seam[row]], pixels[row][seam[row]+1:]), axis=0)

	return Image.fromarray(np.ubyte(arr))


def save_seamed(image, seam, filename, arr): #! DEBUG: arr
	mark_seam(image, seam, arr) #! DEBUG: arr
	save_image(image, filename)


def save_image(image, filename):
	image.save(filename)


def carve(
	in_filename="broadway_tower.jpg",
	out_filename="output/optimized/broadway_tower",
	iteration_count=1
	):

	image = Image.open(in_filename)

	with Pool(processes=50) as pool:
		for i in range(iteration_count):
			print("Processing", (i+1), "/", iteration_count)
			seam_filename = f"{out_filename}_{str(i).zfill(4)}_seamed.png"
			shrunk_filename = f"{out_filename}_{str(i).zfill(4)}_shrunk.png"

			arr = np.array(image, dtype=np.longlong)
			darr = get_distances(arr) #! DEBUG
			seam = get_seam(darr.copy())
			
			# mark seam and save
			res = pool.apply_async(save_seamed, (image, seam, seam_filename, darr)) #! DEBUG: arr
			print(res.get())
			
			# remove seam and save
			shrunk = remove_seam(arr, seam)
			pool.apply_async(save_image, (shrunk, shrunk_filename))
			image = shrunk
		pool.close()
		print("Saving ...")
		pool.join()

	print("Done :)")

if __name__ == "__main__":
	carve()

