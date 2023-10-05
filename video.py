import os
import glob
import cv2
from tqdm import tqdm
def images_to_video(image_folder, video_name, fps):
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

# Thay đổi 'hdnerv2/dnerv' thành đường dẫn thư mục chứa ảnh của bạn
# images_to_video('merge_images/dnerv/Beauty', 'output.mp4', 30)


if __name__=='__main__':
    in_dir = 'merge_images'
    out_dir = 'video'
    model_list = ['gt', 'dnerv', 'hdnerv2', 'hdnerv3']
    video_list = ["Bosphorus","YachtRide", "HoneyBee", "ShakeNDry", "Jockey", "Beauty", "ReadySteadyGo"]

    for model in tqdm(model_list):
        in_model_path = os.path.join(in_dir, model)
        out_model_path = os.path.join(out_dir, model)
        for video in video_list:
            in_video_dir = os.path.join(in_model_path, video)
            out_video_dir = os.path.join(out_model_path, '{}.mp4'.format(video))
            images_to_video(in_video_dir, out_video_dir, 30)
    
    

        