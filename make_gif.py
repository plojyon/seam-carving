import sys
from tqdm import tqdm
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
	size = Image.open(frame_paths[0]).size

	fourcc = cv2.VideoWriter_fourcc(*'mp4v') #(*'avc1')
	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	video = cv2.VideoWriter(output_path, fourcc, 60, size)
	for path in tqdm(frame_paths):
		image = Image.open(path)
		frame = Image.new("RGB", size)
		width_diff = size[0] - image.size[0]
		frame.paste(image, (width_diff//2, 0))
		video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
	video.release()

# make_gif(["broadway_tower.jpg", "output/shrinkage/broadway_tower_0080_shrunk.png"], "output/comparison.gif")
frames = ["broadway_tower.jpg"] + [f"output/optimized/broadway_tower_{str(i//2).zfill(4)}_{'seamed' if i % 2 == 0 else 'shrunk'}.png" for i in range(2000)]
make_video(frames, "output/comparison.mp4")
