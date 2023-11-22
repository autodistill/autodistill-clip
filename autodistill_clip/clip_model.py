import os
from dataclasses import dataclass

import sys
from PIL import Image
import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology
from autodistill.classification import ClassificationBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CLIP(ClassificationBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        if not os.path.exists(f"{HOME}/.cache/autodistill/clip"):
            os.makedirs(f"{HOME}/.cache/autodistill/clip")

            os.system("pip install ftfy regex tqdm")
            os.system(f"cd {HOME}/.cache/autodistill/clip && pip install git+https://github.com/openai/CLIP.git")

        # add clip path to path
        sys.path.insert(0, f"{HOME}/.cache/autodistill/clip/CLIP")

        import clip

        model, preprocess = clip.load("ViT-B/32", device=DEVICE)

        self.clip_model = model
        self.clip_preprocess = preprocess
        self.tokenize = clip.tokenize

    def embed_image(self, input: str) -> np.ndarray:
        image = self.clip_preprocess(Image.open(input)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        return image_features.cpu().numpy()
    
    def embed_text(self, input: str) -> np.ndarray:
        return self.clip_model.encode_text(self.tokenize([input]).to(DEVICE)).cpu().numpy()

    def predict(self, input: str) -> sv.Classifications:
        labels = self.ontology.prompts()

        image = self.clip_preprocess(Image.open(input)).unsqueeze(0).to(DEVICE)

        if isinstance(labels[0], str):
            text = self.tokenize(labels).to(DEVICE)

            with torch.no_grad():
                logits_per_image, _ = self.clip_model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                top_prob = np.max(probs)
                top_class = np.argmax(probs)
        else:
            # each label is an embedding, do cosine sim
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)

            similarities = []

            for label in labels:
                label_features = torch.tensor(np.array(label).reshape(1, -1)).to(DEVICE)
                similarity = torch.cosine_similarity(image_features, label_features)
                similarities.append(similarity.item())

            top_prob = np.max(similarities)
            top_class = np.argmax(similarities)

        return sv.Classifications(
            class_id=np.array([top_class]),
            confidence=np.array([top_prob]),
        )