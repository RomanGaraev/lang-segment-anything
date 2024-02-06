import os
import re
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

from pelper import print_duration


CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    no_dino_resize = bool(int(os.getenv("NO_DINO_RESIZE", "0")))
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) if not no_dino_resize else T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


def filter_by_constraints(boxes, logits, phrases, area_constraints, phrases_constraints, W, H):
    idxs = []
    for i, box in enumerate(boxes):
        phrase = phrases[i]
        if phrases_constraints is not None:
            if phrase not in phrases_constraints:
                print(f"Skipping box: phrase {phrase} it not in {phrases_constraints}")
                continue
        if area_constraints is not None:
            x1,y1,x2,y2 = box.tolist()
            w = (x2 - x1)/W
            h = (y2 - y1)/H
            area = w*h

            min_area, max_area = area_constraints[phrase]
            if area < min_area or area > max_area:
                print(f"Skipping box: area {area} is out of limits [{min_area}, {max_area}] for phrase {phrase}")
                continue
        idxs.append(i)
    return [
        torch.Tensor([boxes[i].tolist() for i in idxs]),
        torch.Tensor([logits[i].tolist() for i in idxs]),
        [phrases[i] for i in idxs]
    ]

def cleanup_phrases(phrases): # [CLS] traffic light utility pole [SEP]
    out = []
    for phrase in phrases:
        phrase = re.sub("\[[A-Z]+\]", "", phrase)
        phrase = phrase.strip()
        out.append(phrase)
    return out

class LangSAM():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    @print_duration("predict_dino")
    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    @print_duration("predict")
    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25, area_constraints=None, phrases_constraints=None):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        phrases = cleanup_phrases(phrases)

        if area_constraints is not None or phrases_constraints is not None:
            W, H = image_pil.size
            boxes, logits, phrases = filter_by_constraints(boxes, logits, phrases, area_constraints, phrases_constraints, W, H)
        return boxes, phrases, logits
