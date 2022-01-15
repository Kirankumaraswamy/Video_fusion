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
import cv2 as cv

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

class UCF11_pretrained(Dataset):
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

    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None, extensions=("avi"), model=None, args=None):
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

        torch.set_grad_enabled(False)
        model.eval()

        self.data =[]
        print("It might take some time to initialize the features from pretrained model for the first time....")
        for i in range(len(self.samples)):
            if i % 200 == 0:
                print("Image {} of {}.".format(i+1, len(self.samples)))
            videoname = self.samples[i][0]
            class_idx = self.samples[i][1]
            videodata = skvideo.io.vread(videoname)
            length, height, width, channel = videodata.shape
            clip_start = random.randint(0, length - self.clip_len)
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

            clip = clip[None,:,:,:,:]

            if args.gpu is not None:
                clip = clip.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            outputs = model(clip)
            outputs = torch.squeeze(outputs)
            self.data.append((outputs.cpu(), torch.tensor(int(class_idx))))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]       """


        return self.data[idx]

class UCF11Motion(Dataset):
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
        self.clip_len = clip_len + 1
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = (self.clip_len) * tuple_len + interval * (tuple_len - 1)

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
        if length < 24:
            print(filename)
        clip_start = random.randint(0, length - (self.clip_len ))
        clip = videodata[clip_start: clip_start + (self.clip_len)]

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

        #clip= clip[:, 1:self.clip_len, :, :] - clip[:, :self.clip_len-1, :, :]

        clips = []
        for i in range(int(self.clip_len - 1)):
            if i % 2 == 0:
                clips.append(clip[:, i + 1, :, :] - clip[:, i, :, :])

        clip = torch.stack(clips).permute([1, 0, 2, 3])

        '''
        clips = []
        # Parameters
        blur = 21
        canny_low = 0.05
        canny_high = 0.58
        min_area = 0.0005
        max_area = 0.95
        dilate_iter = 10
        erode_iter = 10
        mask_color = (0.0, 0.0, 0.0)

        for i in range(self.clip_len):
            frame = clip[:, i, :, :].numpy()
            frame = np.moveaxis(frame, 0, -1).astype('uint8')

            # Convert image to grayscale
            image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Apply Canny Edge Dection
            edges = cv.Canny(image_gray, canny_low, canny_high)

            edges = cv.dilate(edges, None)
            edges = cv.erode(edges, None)

            # get the contours and their areas
            contour_info = [(c, cv.contourArea(c),) for c in
                            cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]]

            # Get the area of the image as a comparison
            image_area = frame.shape[0] * frame.shape[1]

            # calculate max and min areas in terms of pixels
            max_area = max_area * image_area
            min_area = min_area * image_area

            # Set up mask with a matrix of 0's
            mask = np.zeros(edges.shape, dtype=np.uint8)

            # Go through and find relevant contours and apply to mask
            for contour in contour_info:  # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
                if contour[1] > min_area and contour[1] < max_area:
                    # Add contour to mask
                    mask = cv.fillConvexPoly(mask, contour[0], (255))

            # use dilate, erode, and blur to smooth out the mask
            mask = cv.dilate(mask, None, iterations=dilate_iter)
            mask = cv.erode(mask, None, iterations=erode_iter)
            mask = cv.GaussianBlur(mask, (blur, blur), 0)

            # Ensures data types match up
            mask_stack = mask.astype('float32') / 255.0
            frame = frame.astype('float32') / 255.0


            a = mask_stack.dot(frame)
            # Blend the image and the mask
            masked = mask_stack.dot(frame)
            masked = (masked * 255).astype('uint8')
            cv.imshow("Foreground", masked)
            
            
            clip = torch.stack(clips).permute([1, 0, 2, 3])



        a=1

        
        clips = []
        for i in range(int(self.clip_len)):
            clips.append(clip[:, i + 1, :, :] - clip[:, i, :, :])
                # clips.append(clip[:, 1:self.clip_len + 1, :, :] - clip[:, :self.clip_len, :, :])'''



        '''for i in range(3):
            a = clip[:, i, :, :]
            a = a.numpy()
            a = np.moveaxis(a, 0, -1)
            plt.imshow(a)
            plt.show()'''

        return clip, torch.tensor(int(class_idx))