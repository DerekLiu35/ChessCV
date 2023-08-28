"""Class for loading the Chess Positions dataset plus utilities."""
from pathlib import Path
import zipfile

class ChessPositions:
    """A dataset of images of a randomly generated chess positions of 5-15 pieces (2 kings and 3-13 pawns/pieces)

    https://www.kaggle.com/datasets/koryakinp/chess-positions
    """

    def __init__(self):
        DATA_DIRNAME = Path("README.md").resolve().parents[0]
        DATA_DIRNAME / "data"

    def prepare_data(self):
        # if self.xml_filenames:
        #     return
        # filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)  # type: ignore
        filename = "chess-positions.zip"
        DL_DATA_DIRNAME = Path("README.md").resolve().parents[0] / "data"
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)

    def __repr__(self):
        """Print info about the dataset."""
        info = ["Chess Positions Dataset"]
        info.append(f"Total Images: {len(self.xml_filenames)}")
        info.append(f"Total Test Images: {len(self.test_ids)}")
        info.append(f"Total Paragraphs: {len(self.paragraph_string_by_id)}")
        num_lines = sum(len(line_regions) for line_regions in self.line_regions_by_id.items())
        info.append(f"Total Lines: {num_lines}")

        return "\n\t".join(info)

def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    print("Extracting IAM data")
    with zipfile.ZipFile(filename, "r") as zip_file:
      zip_file.extractall()


if __name__ == "__main__":
    pass