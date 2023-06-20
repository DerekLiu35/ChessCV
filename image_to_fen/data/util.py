"""Base Dataset class."""
import os
from random import shuffle
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.utils.data

SequenceOrTensor = Union[Sequence, torch.Tensor]

root = '/content/ChessCV'
train_size = 1000
test_size = 500

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        # root is 'content' in this notebook
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        shuffle(self.imgs)
        self.imgs = self.imgs[:train_size]
    def __getitem__(self, idx):
        # load images 
        img_path = os.path.join(self.root, "train", self.imgs[idx])

        img = Image.open(img_path).convert("RGB").resize((downsample_size, downsample_size))

        # get bounding box coordinates and labels
        
        fen = fen_from_filename(img_path)
        boxes, labels = boxes_labels_from_fen(fen)
        num_objs = len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)


        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels


def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)

def fen_from_filename(filename):
  base = os.path.basename(filename)
  return os.path.splitext(base)[0]

def onehot_from_fen(fen):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if(char in '12345678'):
            output = np.append(
              output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)

    return output

def fen_from_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            idx = np.where(one_hot[j*8 + i]==1)[0][0]
            if(idx == 12):
                output += ' '
            else:
                output += piece_symbols[idx]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

downsample_size = 200
square_size = int(downsample_size/8)
piece_symbols = 'prbnkqPRBNKQ'
def boxes_labels_from_fen(fen):
    boxes = np.empty((0, 4))
    labels = []
    x = 0
    y = 0
    for char in fen:
        if(char in '12345678'):
          x += int(char) * square_size
        elif char == '-':
          y += square_size
          x = 0
        else:
            boxes = np.append(boxes, [[x, y, x + square_size, y + square_size]], axis = 0)
            x += square_size
            idx = piece_symbols.index(char)
            labels.append(12 - idx)

    return boxes, labels

import torch
import torch.utils.data
from PIL import Image

root = '/content/dataset'
train_size = 1000
test_size = 500

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        # root is 'content' in this notebook
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        shuffle(self.imgs)
        self.imgs = self.imgs[:train_size]
    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "train", self.imgs[idx])

        img = Image.open(img_path).convert("RGB").resize((downsample_size, downsample_size))

        # get bounding box coordinates and labels

        fen = fen_from_filename(img_path)
        boxes, labels = boxes_labels_from_fen(fen)
        num_objs = len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)


        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))