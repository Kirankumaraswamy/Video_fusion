from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import os
import torch

import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class UCF50(Dataset):
    """UCF101 dataset for Retrieval. Sample clips for each video. The class index start from 0.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, sample_num, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        extensions = ('mp4')

        class_idx_path = os.path.join(root_dir, 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        train_split_path = os.path.join(root_dir, 'trainlist01.txt')
        self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]

        classes = list(sorted(list_dir(root_dir)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root_dir, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        split = root_dir.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root_dir, 'UCF_50_metadata_{}.pt'.format(split))
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        videoname = self.train_split[idx]
        videoname1 = self.samples[idx][0]

        class_idx = self.samples[idx][1]
        filename = videoname1
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        all_clips = []
        all_idx = []
        for i in np.linspace(self.clip_len / 2, length - self.clip_len / 2, self.sample_num):
            clip_start = int(i - self.clip_len / 2)
            clip = videodata[clip_start: clip_start + self.clip_len]
            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)  # PIL image
                    frame = self.transforms_(frame)  # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            all_clips.append(clip)
            all_idx.append(torch.tensor(int(class_idx)))

        '''shuffle = [0, 1, 2]
        random.shuffle(shuffle)
        shuffled_clips = []
        shuffle_index = []
        shuffled_clips.append(all_clips[shuffle[0]])
        shuffled_clips.append(all_clips[shuffle[1]])
        shuffled_clips.append(all_clips[shuffle[2]])
        shuffle_index.append(torch.tensor(int(shuffle[0])))
        shuffle_index.append(torch.tensor(int(shuffle[1])))
        shuffle_index.append(torch.tensor(int(shuffle[2])))
        a = all_clips[0][:, 0, :, :].numpy()
        a = np.moveaxis(a, 0, -1)
        a = shuffled_clips[0][:, 0, :, :].numpy()
        a = np.moveaxis(a, 0, -1)

        return torch.stack(all_clips), torch.stack(shuffle_index)'''
        return torch.stack(all_clips), torch.stack(all_idx)

class UCF50ClipRetreival(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
        extensions = ('mp4')

        class_idx_path = os.path.join(root_dir, 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        train_split_path = os.path.join(root_dir, 'trainlist01.txt')
        self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]

        classes = list(sorted(list_dir(root_dir)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root_dir, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        split = root_dir.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root_dir, 'UCF_50_metadata_{}.pt'.format(split))
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        videoname = self.samples[idx][0]

        class_idx = self.samples[idx][1]
        filename = videoname
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))

        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)  # PIL image
                    frame = self.transforms_(frame)  # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order)
        #return (tuple_clip, tuple_order)

class UCF50SimSiam(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None, extensions=("avi")):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        classes = list(sorted(list_dir(root_dir)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root_dir, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        split = root_dir.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root_dir, 'UCF_50_metadata_{}.pt'.format(split))
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        videoname = self.samples[idx][0]

        class_idx = self.samples[idx][1]
        filename = videoname
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))

        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)  # PIL image
                    frame = self.transforms_(frame)  # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return (tuple_clip, tuple_order)

class UCF50VCOP(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None, extensions=("avi")):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        classes = list(sorted(list_dir(root_dir)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root_dir, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        split = root_dir.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root_dir, 'UCF_50_metadata_{}.pt'.format(split))
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        videoname = self.samples[idx][0]

        class_idx = self.samples[idx][1]
        filename = videoname
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))

        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)  # PIL image
                    frame = self.transforms_(frame)  # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order)

class UCF11(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None, extensions=("avi")):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        classes = list(sorted(list_dir(root_dir)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root_dir, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        split = root_dir.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root_dir, 'UCF_50_metadata_{}.pt'.format(split))
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        videoname = self.samples[idx][0]

        class_idx = self.samples[idx][1]
        filename = videoname

        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        clip_start = random.randint(0, length - self.clip_len)
        clip = videodata[clip_start: clip_start + self.clip_len]

        if self.transforms_:
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = self.toPIL(frame) # PIL image
                frame = self.transforms_(frame) # tensor [C x H x W]
                trans_clip.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        else:
            clip = torch.tensor(clip)

        return clip, torch.tensor(int(class_idx))

        '''
        """
                Returns:
                    clip (tensor): [channel x time x height x width]
                    class_idx (tensor): class index [0-100]
                """
                videoname = self.samples[idx][0]

                class_idx = self.samples[idx][1]
                filename = videoname

                videodata = skvideo.io.vread(filename)
                length, height, width, channel = videodata.shape

                tuple_clip = []

                # random select tuple for train, deterministic random select for test
                if self.train:
                    tuple_start = random.randint(0, length - self.tuple_total_frames)
                else:
                    random.seed(idx)
                    tuple_start = random.randint(0, length - self.tuple_total_frames)

                clip_start = tuple_start
                # for now use only one clip for training
                for _ in range(1):
                    clip = videodata[clip_start: clip_start + self.clip_len]
                    tuple_clip.append(clip)
                    clip_start = clip_start + self.clip_len + self.interval

                if self.transforms_:
                    trans_tuple = []
                    for clip in tuple_clip:
                        trans_clip = []
                        # fix seed, apply the sample `random transformation` for all frames in the clip
                        seed = random.random()
                        for frame in clip:
                            random.seed(seed)
                            frame = self.toPIL(frame)  # PIL image
                            frame = self.transforms_(frame)  # tensor [C x H x W]
                            trans_clip.append(frame)
                        # (T x C X H x W) to (C X T x H x W)
                        trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                        trans_tuple.append(trans_clip)

                    tuple_clip = trans_tuple
                else:
                    tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

                all_idx  = [(torch.tensor(int(class_idx))) for clip in tuple_clip]

                return tuple_clip[0], torch.tensor(class_idx)
                '''