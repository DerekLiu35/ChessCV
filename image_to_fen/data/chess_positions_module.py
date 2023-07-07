"""IAM Paragraphs Dataset class."""
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from image_to_fen.data.base_data_module import load_and_print_info
from image_to_fen.data.chess_positions import ChessPositions
from image_to_fen.data.util import BaseDataset, resize_image, ChessDataset, get_transform, collate_fn


from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

import lightning as pl

class ChessDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = Path("README.md").resolve().parents[0]):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        DATA_DIRNAME = Path("README.md").resolve().parents[0]
        DATA_DIRNAME / "data"
        if (DATA_DIRNAME / "train").exists():
          return
        # download
        # filename = "chess-positions.zip"
        # DL_DATA_DIRNAME = Path("README.md").resolve().parents[0] / "data"
        # _extract_raw_dataset(filename, DL_DATA_DIRNAME)
        positions = ChessPositions()
        positions.prepare_data()

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "train" or stage is None:
            chess_full = ChessDataset(root=self.data_dir, transforms=get_transform(train=True))
            self.chess_train, self.chess_val = random_split(chess_full, [.8, .2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(self.chess_train, batch_size=32, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.chess_val, batch_size=32, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.chess_test, batch_size=32, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.chess_predict, batch_size=32)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "Chess Positions Dataset\n")
        #     f"Input dims : {self.input_dims}\n"
        #     f"Output dims: {self.output_dims}\n"
        # )
        # if self.data_train is None and self.data_val is None and self.data_test is None:
        #     return basic

        # x, y = next(iter(self.train_dataloader()))
        # xt, yt = next(iter(self.test_dataloader()))
        # data = (
        #     f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
        #     f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
        #     f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        #     f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
        #     f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        # )
        return basic
    
    


def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]

    # Add 2 because of start and end tokens
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2


def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]

    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {"min": min(_get_property_values("num_lines")), "max": max(_get_property_values("num_lines"))},
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"


def _num_lines(label: str) -> int:
    """Return number of lines of text in label."""
    return label.count(NEW_LINE_TOKEN) + 1


if __name__ == "__main__":
    #load_and_print_info(IAMParagraphs)
    pass