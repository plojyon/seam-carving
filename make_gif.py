import sys
from PIL import Image, ImageDraw
import numpy as np
import cv2

def make_gif(frame_paths, output_path):
	master_image = Image.open(frame_paths.pop(0))

	images = [Image.open(path) for path in frame_paths]
	frames = []
	for image in images:
		frame = Image.new("RGB", master_image.size)
		width_diff = master_image.size[0] - image.size[0]
		frame.paste(image, (width_diff//2, 0))
		frames.append(frame)

	master_image.save(output_path, save_all=True, append_images=frames, loop=0, duration=1000)


def make_video(frame_paths, output_path):
	images = [Image.open(path) for path in frame_paths]

	fourcc = cv2.VideoWriter_fourcc(*'mp4v') #(*'avc1')
	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	video = cv2.VideoWriter(output_path, fourcc, 60, images[0].size)
	for image in images:
		frame = Image.new("RGB", images[0].size)
		width_diff = images[0].size[0] - image.size[0]
		frame.paste(image, (width_diff//2, 0))
		video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
	video.release()

# make_gif(["broadway_tower.jpg", "output/shrinkage/broadway_tower_0080_shrunk.png"], "output/comparison.gif")
frames = ["broadway_tower.jpg"] + [f"output/shrinkage/broadway_tower_{str(i).zfill(4)}_shrunk.png" for i in range(500)]
make_video(frames, "output/comparison.mp4")
