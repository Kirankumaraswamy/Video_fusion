import random
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
import torch
import kornia
import torchvision.transforms as transforms

class VideoAugment(object):

    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment = transforms.Compose(
            [
                kornia.augmentation.RandomGrayscale(p=0.2),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.4),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips