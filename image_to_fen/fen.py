import argparse
from pathlib import Path
from typing import Sequence, Union

from PIL import Image
import torch
import torchvision
import numpy as np
import re

import image_to_fen.util as util


STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "image-to-fen"
MODEL_FILE = "model.pt"

class ImageToFen:
    """Takes image of chess board and returns FEN string."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict FEN string for image of chess board."""
        image = image
        if not isinstance(image, Image.Image):
            image = util.read_image_pil(image, grayscale=True)
        image = image.resize((200, 200))
        image = torchvision.transforms.PILToTensor()(image)/255
        pred = self.model([image])[1][0]
        nms_pred = apply_nms(pred, iou_thresh=0.2)
        pred_str = boxes_labels_to_fen(nms_pred['boxes'], nms_pred['labels'])
        return pred_str
    
def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

def boxes_labels_to_fen(boxes, labels, square_size=25):
  boxes = torch.round(boxes / 25) * 25
  eye = np.eye(13)
  one_hot = onehot_from_fen("8-8-8-8-8-8-8-8")
  for i, box in enumerate(boxes):
    x = box[0]
    y = box[1]
    ind = int((x / square_size) + (y / square_size) * 8)
    if (ind >= 64):
      continue
    one_hot[ind] = eye[12 - labels[i]].reshape((1, 13)).astype(int)
  return fen_from_onehot(one_hot)

def onehot_from_fen(fen):
    piece_symbols = 'prbnkqPRBNKQ'
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
    piece_symbols = 'prbnkqPRBNKQ'
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

def main():
    """Run prediction on image."""
    parser = argparse.ArgumentParser(description="Predict FEN string for image of chess board.")
    parser.add_argument("image", type=Path, help="Path to image file.")
    parser.add_argument("--model-path", type=Path, help="Path to model file.")
    args = parser.parse_args()
    image_to_fen = ImageToFen(args.model_path)
    pred = image_to_fen.predict(args.image)
    print(f"Prediction: {pred}")
    
    # image_to_fen/tests/support/boards/phpSrRLQ1.png
if __name__ == "__main__":
    main()