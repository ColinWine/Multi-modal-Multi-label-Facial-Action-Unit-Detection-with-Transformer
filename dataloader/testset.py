import os

import torch
from PIL import Image

import pickle
import torchaudio
import math
from .clip_transforms import *
from torch.utils.data import Dataset
import lmdb


class Aff2TestDataset(Dataset):
    def __init__(self, opt):
        super(Aff2TestDataset, self).__init__()
        self.task = opt.get('task')
        self.opt = opt
        assert self.task in ['ALL', 'EX', 'AU', 'VA']
        self.audio_dir = r'I:\ABAW_LMDB\audio'
        lmdb_label_path = opt['lmdb_label_dir']
        try:
            self.env_image = lmdb.open(os.path.join(lmdb_label_path, '.croped_aligned_jpeg'), create=False, lock=False,
                                       readonly=True)
        except:
            self.env_image = None
        try:
            self.env_mask = lmdb.open(os.path.join(lmdb_label_path, '.croped_aligned_mask'), create=False, lock=False,
                                       readonly=True)
        except:
            self.env_mask = None

        self.clip_len = opt.get('n_frames')
        self.input_size = (112, 112)
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
        self.mask_clip_transform = ComposeWithInvert([NumpyToTensor(),
                                            Normalize(mean=[0.43216, 0.394666, 0.37645, 0.5],
                                                    std=[0.22803, 0.22145, 0.216989, 0.225])])

        split_dict = pickle.load(open(f'{self.opt["cache_dir"]}/split_dict_test_{self.task}.pkl', 'rb'))
        self.time_stamps = split_dict['timestamp']
        self.image_path = split_dict['image_path']
        self.video_db_nr = split_dict['video_db_nr']
        self.test_ids = split_dict['test']
        self.use_mask = self.opt['use_mask']

    def set_clip_len(self, clip_len):
        assert (np.mod(clip_len, 2) == 0)  # clip length should be even at this point
        self.clip_len = clip_len

    def set_modes(self, modes):
        self.modes = modes

    def __getitem__(self, index):
        data = {'Index': index}

        video_id = os.path.dirname(self.image_path[index])
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
                    pass

                if self.use_mask:
                    try:
                        clip[clip_i, :, :, 3] = mask_img
                    except:
                        pass


        data['AU'] = self.get_label(current_frame_path, task='au')
        data['EX'] = self.get_label(current_frame_path, task='ex')
        data['VA'] = self.get_label(current_frame_path, task='va')
        data['clip'] = self.mask_clip_transform(clip)
        data['video_id'] = video_id
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
        key = video_name + '/' + frame_name
        try:
            with self.env_image.begin(write=False) as txn:
                jpeg = np.frombuffer(txn.get(key.encode()), dtype='uint8')
                image = self.__decodejpeg(jpeg)
                return image
        except:
            # print('No image for:', key)
            return None

    def get_mask(self, video_frame):
        video_name = os.path.dirname(video_frame)
        frame_name = os.path.basename(video_frame)
        key = video_name + '/' + frame_name
        try:
            with self.env_mask.begin(write=False) as txn:
                jpeg = np.frombuffer(txn.get(key.encode()), dtype='uint8')
                mask = self.__decodemask(jpeg)
                return mask
        except:
            # print('No image for:', key)
            return None

    def get_audio_feature(self, video_id, index):
        # get audio
        video_id = video_id.replace('_left', '').replace('_right', '').replace('_main', '')
        audio_file = os.path.join(self.audio_dir, video_id + '.wav')

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
        if task.lower() == 'au':
            return -1 * np.ones(12, dtype=np.int8)
        elif task.lower() == 'ex':
            return -1 * np.ones(1, dtype=np.int8)
        elif task.lower() == 'va':
            return -5 * np.ones(2, dtype=np.float32)

        return -1

    def __len__(self):
        return len(self.image_path)
