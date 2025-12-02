from typing import List

import torch
import torch.nn as nn
import open_clip


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def text_dim(self) -> int:
        return self.model.text_projection.shape[1]

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
        return text_features

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)
