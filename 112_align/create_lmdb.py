import os.path
import glob
from natsort import natsorted
import pickle
import lmdb
import cv2
import numpy as np
from tqdm import tqdm


def yield_buffer(video_path_list, buffer_size=int(1e5)):
    item_idx = 0
    buffer = list()
    for i, v in enumerate(tqdm(video_path_list, desc="Creating Cache")):
        images = glob.glob(os.path.join(v, '*.jpg'))
        v_name = os.path.basename(v)
        bar = tqdm(natsorted(images), desc=f'Creating Cache, {i}, {v_name}', leave=False, position=0, colour='green')
        for img in bar:
            file_name = os.path.basename(img)
            key = f'{v_name}/{file_name}'.encode()
            with open(img, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            buffer.append((key, jpeg.tobytes()))
            # img_data = cv2.imread(img, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            # buffer.append((key, img_data.tobytes()))
            item_idx += 1
            if item_idx % buffer_size == 0:
                yield buffer
                buffer.clear()
    yield buffer


def create_image_cache(src_path, save_path, map_size, buffer_size=int(1e5)):
    os.makedirs(save_path, exist_ok=True)
    video_dirs = glob.glob(os.path.join(src_path, '*'))
    env = lmdb.open(save_path, map_size=map_size)
    for buffer in yield_buffer(video_dirs, buffer_size=buffer_size):
        with env.begin(write=True) as txn:
            for key, value in buffer:
                txn.put(key, value)
    keys_cache_file = os.path.join(save_path, '_keys_cache.p')
    env = lmdb.open(save_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        print('Create lmdb keys cache: {}'.format(keys_cache_file))
        keys = [key.decode() for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    print('Finish creating lmdb keys cache.')


if __name__ == '__main__':
    image_root = r'K:\ABAW2022\data\112_align\cropped_aligned'
    lmdb_save_path = os.path.join(r'K:\ABAW2022\data\112_align', 'lmdb\.croped_jpeg')  # must end with .lmdb

    num_image = 0
    # image_nbyte = 37632
    image_nbyte = 6000
    data_size = 0
    videos = glob.glob(os.path.join(image_root, '*'))
    for v in tqdm(videos, desc='Counting image'):
        images = glob.glob(os.path.join(v, '*.jpg'))
        num_image += len(images)
    data_size = int(image_nbyte * num_image * 1.4)
    print(f"Read done. num_image: {num_image}, data size:{data_size / 1073741824} Gbyte")
    create_image_cache(src_path=image_root, save_path=lmdb_save_path,
                       map_size=data_size, buffer_size=int(1e4))
