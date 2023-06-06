import os
from dataclasses import dataclass

import clip
import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32", device=device)


@dataclass
class CLIP(DetectionBaseModel):
    ontology: CaptionOntology
    clip_model: model

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.clip_model = model

    def predict(self, input: str) -> sv.Detections:
        labels = self.ontology.prompts()

        image = preprocess(cv2.imread(input)).unsqueeze(0).to(DEVICE)
        text = clip.tokenize(labels).to(DEVICE)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            top_k = 1

            values, indices = probs.topk(top_k)

        return sv.Detections(
            xyxy=np.array([[0, 0, 0, 0]]),
            class_ids=np.array(values),
            scores=np.array(indices),
        )
