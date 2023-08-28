"""Chess Dataset class."""
from pathlib import Path

from image_to_fen.data.chess_positions import ChessPositions
from image_to_fen.data.util import ChessDataset, get_transform, collate_fn


from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision import transforms

import pytorch_lightning as pl

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
        positions = ChessPositions()
        positions.prepare_data()

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "train" or stage is None:
            chess_full = ChessDataset(root=self.data_dir, transforms=get_transform(train=True))
            self.chess_train, self.chess_val = random_split(chess_full, [.8, .2])

    def train_dataloader(self):
        return DataLoader(self.chess_train, batch_size=32, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.chess_val, batch_size=32, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.chess_test, batch_size=32, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.chess_predict, batch_size=32)

if __name__ == "__main__":
    pass