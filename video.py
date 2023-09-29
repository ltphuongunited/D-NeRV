import threading
from tqdm import tqdm
import torch
from PIL import Image
import os
import numpy as np


def split_list(l, n):
    """Yield successive n-sized chunks from l."""
    length = len(l)
    chunk_size = round(length / n)
    for i in range(0, length, chunk_size):
        yield l[i:i + chunk_size]

def merge_images(video_name, video_length, gt_base_dir):
    h = 1024
    w = 1920
    split_size_h = 256
    split_size_w = 320
    split_num_h = h // split_size_h
    split_num_w = w // split_size_w

    folder = os.path.join('merge_images/gt', video_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    for frame_index in range(video_length):

        gt_image_list = []
        for i in range(1, split_num_h * split_num_w + 1):
            gt_image = Image.open(os.path.join(gt_base_dir, "{}-{:02d}".format(video_name, i), 'frame{:06}.png'.format(frame_index + 1))).convert("RGB")
            gt_image_list.append(np.array(gt_image).astype(np.uint8))
            gt_image.close()
        # combine the split 256x320 frame patches into 1024x1920 full frame
        gt_image = np.stack(gt_image_list, axis=0)
        gt_image = gt_image.reshape(split_num_h, split_num_w, split_size_h, split_size_w, 3)
        gt_image = gt_image.transpose(0, 2, 1, 3, 4).reshape(h, w, 3)
        
        gt_image_pil = Image.fromarray(gt_image)
        # Lưu hình ảnh dưới dạng .png
        image_path = os.path.join(folder, "{:03d}.png".format(frame_index))

        gt_image_pil.save(image_path)




if __name__ == "__main__":
    video_length_list = [["Bosphorus", 600], ["YachtRide", 600], ["HoneyBee", 600], ["ShakeNDry", 300], ["Jockey", 600], ["Beauty", 600], ["ReadySteadyGo", 600]]
    gt_base_dir = 'data/UVG/gt'
    NUM_THREADS = 4
    splits = list(split_list(video_length_list, NUM_THREADS))

    def target(video_list):
        for video, video_length in tqdm(video_list):
            merge_images(video, video_length, gt_base_dir)

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()