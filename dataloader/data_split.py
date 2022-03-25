from tqdm import tqdm
import numpy as np
import pickle
import torchaudio
import subprocess
from .utils import *
from .clip_transforms import *
from .video import Video
from natsort import natsorted
from collections import defaultdict


def create_dataset_split(database_path, save_dir=None):
    all_videos = find_all_video_files(database_path)
    split_dict = defaultdict(defaultdict)

    for v in ['AU', 'EX', 'VA', 'ALL']:
        split_dict[v]['train'] = []
        split_dict[v]['val'] = []
        split_dict[v]['test'] = []
        split_dict[v]['timestamp'] = []
        split_dict[v]['image_path'] = []

    extracted_dir = os.path.join(database_path, 'extracted')
    for video in tqdm(all_videos):
        meta = Video(video, write=False).meta
        meta['filename'] = get_filename(video)
        meta['path'] = get_path(video)
        meta['extension'] = get_extension(video)
        num_frames_video = meta['num_frames']
        frame_dir = os.path.join(extracted_dir, meta['filename'])
        audio_file = os.path.splitext(video)[0] + '.wav'
        ad = torchaudio.info(audio_file)
        assert ad.sample_rate == 44100
        video_ts_file = os.path.join(meta['path'], meta['filename'] + '_video_ts.txt')
        if os.path.isfile(video_ts_file):
            pass
        else:
            mkvfile = os.path.join(meta['path'], 'temp.mkv')
            videofile = os.path.join(meta['path'], meta['filename'] + meta['extension'])
            command = 'mkvmerge -o ' + mkvfile + ' ' + videofile
            subprocess.call(command, shell=True)
            command = 'mkvextract ' + mkvfile + ' timestamps_v2 0:' + video_ts_file
            subprocess.call(command, shell=True)
            os.remove(mkvfile)
        with open(video_ts_file, 'r') as f:
            time_stamps = np.genfromtxt(f)[:num_frames_video]

        split_all = []
        split_au = []
        split_ex = []
        split_va = []
        if 'AU' in meta:
            au_split = meta['AU']
            split_all.append(au_split)
            split_au.append(au_split)
        if 'EX' in meta:
            ex_split = meta['EX']
            split_all.append(ex_split)
            split_ex.append(ex_split)
        if 'VA' in meta:
            va_split = meta['VA']
            split_all.append(va_split)
            split_va.append(va_split)

        split_all = list(set(split_all))  # UPDATED 03.06.2020 (was missing)
        split_au = list(set(split_au))
        split_ex = list(set(split_ex))
        split_va = list(set(split_va))

        real_time_stamp = []
        isTimestampsCreated = False
        for split in split_all:
            for image_filename in natsorted(os.listdir(frame_dir)):
                if not image_filename.endswith('.jpg'):
                    continue
                if os.path.isdir(os.path.join(frame_dir, image_filename)):
                    continue
                split_dict['ALL']['image_path'].append(
                    os.path.relpath(os.path.join(frame_dir, image_filename), extracted_dir))

                if not isTimestampsCreated:
                    index = int(image_filename.split('.')[0]) - 1
                    try:
                        real_time_stamp.append(time_stamps[index])
                    except IndexError as e:
                        # print('\n', e)
                        real_time_stamp.append(time_stamps[-1])

                split_dict['ALL']['train'].append(1 if split == 'train' else 0)
                split_dict['ALL']['val'].append(1 if split == 'val' else 0)
                split_dict['ALL']['test'].append(1 if split == 'test' else 0)
                if split in split_ex:
                    split_dict['EX']['train'].append(1 if split == 'train' else 0)
                    split_dict['EX']['val'].append(1 if split == 'val' else 0)
                    split_dict['EX']['test'].append(1 if split == 'test' else 0)
                if split in split_au:
                    split_dict['AU']['train'].append(1 if split == 'train' else 0)
                    split_dict['AU']['val'].append(1 if split == 'val' else 0)
                    split_dict['AU']['test'].append(1 if split == 'test' else 0)
                if split in split_va:
                    split_dict['VA']['train'].append(1 if split == 'train' else 0)
                    split_dict['VA']['val'].append(1 if split == 'val' else 0)
                    split_dict['VA']['test'].append(1 if split == 'test' else 0)
            isTimestampsCreated = True
            split_dict['ALL']['timestamp'].append(real_time_stamp)
            if split in split_ex:
                split_dict['EX']['timestamp'].append(real_time_stamp)
            if split in split_au:
                split_dict['AU']['timestamp'].append(real_time_stamp)
            if split in split_va:
                split_dict['VA']['timestamp'].append(real_time_stamp)
    if isinstance(save_dir, str):
        os.makedirs(save_dir, exist_ok=True)
        split_dict = create_dataset_split(database_path)
        for v in ['AU', 'EX', 'VA', 'ALL']:
            split_dict[v]['timestamp'] = np.hstack(split_dict[v]['timestamp'])
            video_db_nr = np.hstack(
                [i * np.ones(len(v), dtype=np.int) for i, v in enumerate(split_dict[v]['timestamp'])])
            split_dict[v]['video_db_nr'] = video_db_nr
            split_dict[v]['timestamp'] = np.hstack(split_dict[v]['timestamp'])
            pickle.dump(split_dict[v], open(os.path.join(save_dir, f'split_dict_{v}.pkl'), 'wb'))
    return split_dict


if __name__ == '__main__':
    database_path = r'E:\Datasets\ABAW2\alignmentdata\aff2_processed/'
    save_dir = '../data'
    os.makedirs(save_dir, exist_ok=True)

    create_dataset_split(database_path, save_dir=save_dir)
