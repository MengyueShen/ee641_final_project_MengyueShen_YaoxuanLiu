import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoTextImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: int = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "val"]
        self.root = root
        self.split = split
        self.transform = transform

        ann_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
        img_dir = os.path.join(root, f"{split}2017")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds()
        if max_samples is not None:
            img_ids = img_ids[: max_samples]
        self.samples = []

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info["file_name"])
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if not anns:
                continue
            caption = anns[0]["caption"]
            if not os.path.exists(img_path):
                continue
            self.samples.append(
                {
                    "image_path": img_path,
                    "caption": caption,
                    "image_id": img_id,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = item["caption"]
        return image, caption


def coco_collate_fn(batch: List[Tuple[torch.Tensor, str]]):
    images = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    return images, texts

