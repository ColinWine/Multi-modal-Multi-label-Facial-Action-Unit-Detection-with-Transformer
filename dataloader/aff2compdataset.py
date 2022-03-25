import os

import torch
from PIL import Image

import pickle
import torchaudio
import math
from .clip_transforms import *
from .autoaugment import ImageNetPolicy
from torch.utils.data import Dataset
import lmdb
from .data_split import create_dataset_split


class Aff2CompDataset(Dataset):
    def __init__(self, opt):
        super(Aff2CompDataset, self).__init__()
        self.task = opt.get('task')
        self.opt = opt
        assert self.task in ['ALL', 'EX', 'AU', 'VA']
        self.video_dir = opt.get('root')
        lmdb_label_path = opt['lmdb_label_dir']
        self.extracted_dir = os.path.join(self.video_dir, 'extracted')
        try:
            self.env_image = lmdb.open(os.path.join(lmdb_label_path, '.croped_jpeg'), create=False, lock=False,
                                       readonly=True)
        except:
            print('fail to open image lmdb')
            self.env_image = None
        try:
            self.env_mask = lmdb.open(os.path.join(lmdb_label_path, '.croped_mask'), create=False, lock=False,
                                       readonly=True)
        except:
            self.env_mask = None
            print('fail to open mask lmdb')
        self.env_au = lmdb.open(os.path.join(lmdb_label_path, '.label_au'), create=False, lock=False, readonly=True)
        self.env_ex = lmdb.open(os.path.join(lmdb_label_path, '.label_expr'), create=False, lock=False, readonly=True)
        self.env_va = lmdb.open(os.path.join(lmdb_label_path, '.label_va'), create=False, lock=False, readonly=True)
        self.video2orignal = pickle.load(open(os.path.join(self.video_dir, 'video2orignal.pkl'), 'rb'))

        self.clip_len = opt.get('n_frames')
        self.input_size = (opt.get('image_size'), opt.get('image_size'))
        self.dilation = opt.get('dilation')
        self.label_frame = self.clip_len * self.dilation

        # audio params
        self.window_size = 20e-3
        self.window_stride = 10e-3
        self.sample_rate = 44100
        num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
        window_fn = torch.hann_window

        self.sample_len_secs = opt['audio_len_secs']
        self.sample_len_frames = self.sample_len_secs * self.sample_rate
        self.audio_shift_sec = opt['audio_shift_secs']
        self.audio_shift_samples = self.audio_shift_sec * self.sample_rate
        # transforms

        self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=opt['n_mels'],
                                                                    n_fft=num_fft,
                                                                    win_length=int(self.window_size * self.sample_rate),
                                                                    hop_length=int(self.window_stride
                                                                                   * self.sample_rate),
                                                                    window_fn=window_fn)

        self.audio_spec_transform = ComposeWithInvert([AmpToDB(),
                                                       Normalize(mean=[-14.8], std=[19.895])])
        self.clip_transform = ComposeWithInvert([NumpyToTensor(),
                                                 Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                           std=[0.22803, 0.22145, 0.216989])])
        self.aug_clip_transform = ComposeWithInvert([ImageNetPolicy(),RandomClipFlip(),NumpyToTensor(),
                                                 Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                           std=[0.22803, 0.22145, 0.216989])])
        self.mask_clip_transform = ComposeWithInvert([RandomClipFlip(),NumpyToTensor(),
                                            Normalize(mean=[0.43216, 0.394666, 0.37645, 0.5],
                                                    std=[0.22803, 0.22145, 0.216989, 0.225])])

        # all_videos = find_all_video_files(self.video_dir)
        self.cached_metadata_path = os.path.join(opt['cache_dir'], f'split_dict_{self.task}.pkl')

        if not os.path.isfile(self.cached_metadata_path):
            print('creating cached_metadata... ')
            split_dict = create_dataset_split(self.video_dir, save_dir=self.opt['cache_dir'])
            self.time_stamps = split_dict['timestamp']
            self.image_path = split_dict['image_path']
            self.train_ids = split_dict['train']
            self.val_ids = split_dict['val']
            self.video_db_nr = split_dict['video_db_nr']
        else:
            split_dict = pickle.load(open(f'{self.opt["cache_dir"]}/split_dict_{self.task}.pkl', 'rb'))
            self.time_stamps = split_dict['timestamp']
            self.image_path = split_dict['image_path']
            self.train_ids = split_dict['train']
            self.val_ids = split_dict['val']
            self.video_db_nr = split_dict['video_db_nr']
        self.use_mask = False
        if 'M' in self.opt['modality']:
            self.use_mask = True
            print('use mask')

        self.set_aug(False)

    def set_clip_len(self, clip_len):
        assert (np.mod(clip_len, 2) == 0)  # clip length should be even at this point
        self.clip_len = clip_len

    def set_modes(self, modes):
        self.modes = modes

    def set_aug(self,aug):
        self.aug =aug

    def __getitem__(self, index):
        data = {'Index': index}
        #print(self.image_path[index])
        video_id = os.path.dirname(self.image_path[index])
        #print(self.video2orignal[video_id])
        video_db_nr = self.video_db_nr[index]
        current_frame_path = self.image_path[index]
        if self.use_mask:
            clip = np.zeros((self.clip_len, self.input_size[0], self.input_size[1], 4), dtype=np.uint8)
            # init all frames black
        else:
            clip = np.zeros((self.clip_len, self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        _range = range(index - self.label_frame + self.dilation,
                       index - self.label_frame + self.dilation * (self.clip_len + 1), self.dilation)
        for clip_i, all_i in enumerate(_range):
            if all_i < 0 or all_i >= len(self) or self.video_db_nr[all_i] != video_db_nr:
                # leave frame black
                # video ids that might be the same for different videos
                continue
            else:
                if self.env_image is not None:
                    img = self.get_image(self.image_path[all_i])
                    #img = img.reshape(self.input_size[0], self.input_size[1], 3)
                else:
                    img = np.array(Image.open(os.path.join(self.extracted_dir, self.image_path[all_i])))
                if self.use_mask:
                    if self.env_mask is not None:
                        mask_img = self.get_mask(self.image_path[all_i])
                try:
                    clip[clip_i, :, :, 0:3] = img
                except:
                    # loading an image fails leave that frame black
                    #print('black frame: ',os.path.join(self.extracted_dir, self.image_path[all_i]))
                    pass

                if self.use_mask:
                    try:
                        clip[clip_i, :, :, 3] = mask_img
                    except:
                        # loading an image fails leave that frame black
                        #print('black mask: ',os.path.join(self.extracted_dir, self.image_path[all_i]))
                        pass


        data['AU'] = self.get_label(current_frame_path, task='au')
        data['EX'] = self.get_label(current_frame_path, task='ex')
        data['VA'] = self.get_label(current_frame_path, task='va')

        if not self.use_mask:
            if self.aug:
                data['clip'] = self.aug_clip_transform(clip)
            else:
                data['clip'] = self.clip_transform(clip)
        else:
            data['clip'] = self.mask_clip_transform(clip)
            

        if 'A' in self.opt['modality'].split(';'):
            audio_features, audio = self.get_audio_feature(video_id, index)
            data['audio_features'] = audio_features
            data['audio'] = audio
        return data

    def __decodejpeg(self, jpeg):
        x = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def __decodemask(self, jpeg):
        x = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
        return x

    def get_image(self, video_frame):
        video_name = os.path.dirname(video_frame)
        frame_name = os.path.basename(video_frame)
        video_name = self.video2orignal[video_name]
        key = video_name + '/' + frame_name
        try:
            with self.env_image.begin(write=False) as txn:
                jpeg = np.frombuffer(txn.get(key.encode()), dtype='uint8')
                image = self.__decodejpeg(jpeg)
                return image
        except:
            #print('No image for:', key)
            return None

    def get_mask(self, video_frame):
        video_name = os.path.dirname(video_frame)
        frame_name = os.path.basename(video_frame)
        video_name = self.video2orignal[video_name]
        key = video_name + '/' + frame_name
        try:
            with self.env_mask.begin(write=False) as txn:
                jpeg = np.frombuffer(txn.get(key.encode()), dtype='uint8')
                mask = self.__decodemask(jpeg)
                return mask
        except:
            #print('No image for:', key)
            return None 

    def get_audio_feature(self, video_id, index):
        # get audio
        audio_file = os.path.join(self.video_dir, video_id + '.wav')

        audio, sample_rate = \
            torchaudio.load(audio_file,
                            num_frames=min(self.sample_len_frames,
                                           max(int(
                                               (self.time_stamps[index] / 1000) * self.sample_rate),
                                               int(self.window_size * self.sample_rate))),
                            offset=max(int((self.time_stamps[index] / 1000) * self.sample_rate
                                                 - self.sample_len_frames + self.audio_shift_samples),
                                             0))
        try:
            audio_features = self.audio_transform(audio).detach()
        except Exception as e:
            print('\n', e, audio_file, self.time_stamps[index] / 1000, self.image_path[index])
            audio = torch.zeros(1, self.sample_len_frames)
            audio_features = self.audio_transform(audio).detach()

        if audio.shape[1] < self.sample_len_frames:
            _audio_features = torch.zeros((audio_features.shape[0], audio_features.shape[1],
                                           int((self.sample_len_secs / self.window_stride) + 1)))
            _audio_features[:, :, -audio_features.shape[2]:] = audio_features
            audio_features = _audio_features

        if self.audio_spec_transform is not None:
            audio_features = self.audio_spec_transform(audio_features)

        if audio.shape[1] < self.sample_len_frames:
            _audio = torch.zeros((1, self.sample_len_frames))
            _audio[:, -audio.shape[1]:] = audio
            audio = _audio
        return audio_features, audio

    def get_label(self, video_frame, task):
        """

        Args:
            video_frame: like '001_AU1v_EX1__VA1v/00001.jpg'
            task: one of [au, ex, va]

        Returns:

        """
        video_name = os.path.dirname(video_frame)
        frame_name = os.path.basename(video_frame)
        video_name = self.video2orignal[video_name]
        key = video_name + '/' + frame_name

        if task.lower() == 'au':
            try:
                with self.env_au.begin(write=False) as txn:
                    label = np.frombuffer(txn.get(key.encode()), dtype=np.int8)
                    return label
            except:
                # print('No AU label for:', key)
                return -1 * np.ones(12, dtype=np.int8)
        elif task == 'ex':
            try:
                with self.env_ex.begin(write=False) as txn:
                    label = np.frombuffer(txn.get(key.encode()), dtype=np.int8)
                    return label
            except:
                # print('No EX label for:', key)
                return -1 * np.ones(1, dtype=np.int8)
        elif task == 'va':
            try:
                with self.env_va.begin(write=False) as txn:
                    label = np.frombuffer(txn.get(key.encode()), dtype=np.float32)
                    return label
            except:
                # print('No VA label for:', key)
                return -5 * np.ones(2, dtype=np.float32)

        return -1

    def __len__(self):
        return len(self.image_path)
