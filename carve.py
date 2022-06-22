import sys
import math
from PIL import Image


def distance(px1, px2):
	"""Euclidean distance between two pixels."""
	return sum([(a-b)**2 for a,b in zip(px1, px2)])


def index_seams(pixels, width, height, progress=True):
	"""Return a width*height array of (next_x, next_y, seam_cost)."""
	seams = [[[0, 0, 0] for _ in range(width)] for __ in range(height)]

	for row in range(height-2, -1, -1):
		if (progress and row % 100 == 0):
			print(100-((row*100)//height), "%")
		
		for col in range(width):
			this_pixel = pixels[col, row]

			candidates = [(row+1, col)]
			if col != 0:
				candidates.append((row+1, col-1))
			if col != width-1:
				candidates.append((row+1, col+1))

			distances = [distance(this_pixel, c) for c in candidates]
			
			idx = distances.index(min(distances)) 
			seams[row][col] = [candidates[idx][0], candidates[idx][1], distances[idx]]

	return seams


def get_seam(seams_index):
	"""Return a list of column indices so that seam[row]=seam_column."""
	height = len(seams_index)
	seam_start_column = seams_index[0].index(min(seams_index[0], key=lambda x: x[2]))
	seam = [seam_start_column]
	current = (0, seam_start_column)
	while (current[0] != height-1):
		current = seams_index[current[0]][current[1]]
		seam.append(current[1])
	return seam


def mark_seam(pixels, seam):
	"""Colour all pixels in a seam, given an array as returned by get_seam."""
	mark_colour = (255, 255, 255) if MODE == "RGB" else (255, 255, 255, 255)
	for row in range(len(seam)):
		pixels[seam[row], row] = mark_colour


def remove_seam(pixels, seam, width, height):
	"""Creates a copy of a bitmap without the seam."""
	shrunk = Image.new(MODE, (width-1, height))
	shrunk_px = shrunk.load()
	for row in range(height):
		offset = 0
		for col in range(width):
			if seam[row] == col:
				offset = 1
				continue
			shrunk_px[col-offset, row] = pixels[col, row]
	return shrunk


if __name__ == "__main__":
	MODE = "RGB"

	if len(sys.argv) > 1:
		filename = sys.argv[1]
	else:
		filename = "broadway_tower.jpg"

	image = Image.open(filename).convert("RGB")

	iteration_count = 1
	for i in range(iteration_count):
		print("Processing", (i+1), "/", iteration_count)

		pixels = image.load()
		seam = get_seam(index_seams(pixels, *image.size))

		mark_seam(pixels, seam)
		image.save("shrinkage/" + filename[:-4] + "_" + str(i).zfill(4) + "_seamed.png")

		shrunk = remove_seam(pixels, seam, *image.size)
		shrunk.save("shrinkage/" + filename[:-4] + "_" + str(i).zfill(4) + "_shrunk.png")

		image = shrunk

	print("Done :)")
