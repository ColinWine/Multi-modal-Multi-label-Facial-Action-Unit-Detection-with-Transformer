import glob
import pickle

import numpy as np
from dataloader.video import Video
import os
import natsort
from tqdm import tqdm

audio_path = r'E:\Datasets\ABAW2\audio'
video_path = r'E:\Datasets\ABAW2\video/all'
crop_alined_path = r'E:\Datasets\ABAW2\cropped_aligned'
prediction_path = r'E:\PyProjects\compete\ABAW2020TNT\result2'
new_dir = 'prediction_new3'
videos = os.listdir(video_path)

# video_dict = dict()
# for v in tqdm(videos):
#     if not v.endswith('.mp4') and not v.endswith('.avi'):
#         continue
#     video_file = os.path.join(video_path, v)
#     video = Video(video_file)
#     n_frame = int(video.num_frames)
#     video_dict.update({v:n_frame})
# print(video_dict)
# print(len(video_dict))
# pickle.dump(video_dict, open('n_video_frames.pkl', 'wb'))
video_dict = pickle.load(open('n_video_frames.pkl', 'rb'))
def nearest_interp(source_list, target_len):
    L = len(source_list)
    if L > target_len:
        print(f'L：{L}, target_len：{target_len}')
        print(source_list)
    source_list = sorted(source_list)
    if target_len <= len(source_list):
        return list(range(L))
    out = []
    index = 0
    for i in range(target_len):
        try:
            for _ in range(source_list[index + 1] - source_list[index]):
                out.append(index)
            index += 1
        except:
            for _ in range(target_len - len(out)):
                out.append(index)

    return out


for task in ['AU', 'EXPR', 'VA']:
    prediction_files = glob.glob(os.path.join(prediction_path, task) + '/*.txt')
    for pf in tqdm(prediction_files):
        basename = os.path.basename(pf)
        filename = basename.split('.')[0]
        aligned_name = filename
        dirname = os.path.dirname(pf)
        filename = filename.replace('_main', '').replace('_left', '').replace('_right', '')
        if filename + '.mp4' in videos:
            filename += '.mp4'
        elif filename + '.avi' in videos:
            filename += '.avi'
        else:
            raise Exception(f'No such file:{filename}')
        # video_file = os.path.join(video_path, filename)
        # video = Video(video_file)
        # n_frame = int(video.num_frames)
        n_frame = video_dict[filename]
        # n_frame = video.count_frames()
        # video.release()
        #
        # if n_frame2 != n_frame:
        #     print(filename, n_frame, n_frame2)
        crop_alined_dir = os.path.join(crop_alined_path, aligned_name)
        frames = natsort.natsorted(os.listdir(crop_alined_dir))
        # print(crop_alined_dir)
        frames = [int(frame.split('.')[0]) for frame in frames if '.jpg' in frame]
        with open(pf, 'r') as f:
            pred = f.readlines()
        os.makedirs(f'{new_dir}/{task}', exist_ok=True)
        with open(f'{new_dir}/{task}/{basename}', 'w') as new_f:
            indices = nearest_interp(source_list=frames, target_len=n_frame)
            new_f.write(pred[0])
            assert len(frames) == len(pred) - 1
            for i in range(n_frame):
                index = indices[i] + 1
                p = pred[index]
                new_f.write(p)
            print(n_frame, index, frames[index - 1], len(indices))

# x = [1, 2, 4, 5]
# a = nearest_interp(x, target_len=5)
# print(a)
# print(len(a))
# c = [x[i] for i in a]
# print(c)
