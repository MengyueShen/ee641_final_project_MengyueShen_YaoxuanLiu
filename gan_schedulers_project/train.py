import os
import json
from typing import Dict, List, Tuple

import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.coco_text_image import CocoTextImageDataset, coco_collate_fn
from models.clip_text_encoder import CLIPTextEncoder
from models.generator_discriminator import ConditionalGenerator, ConditionalDiscriminator
from schedulers.annealed import AnnealedScheduler
from schedulers.learnable_monotone import LearnableMonotoneScheduler
from schedulers.adaptive_annealed import AdaptiveAnnealedScheduler
from torchvision.models import inception_v3, Inception_V3_Weights


def apply_augmentation(images: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return images
    if p >= 1.0:
        mask = torch.ones(images.size(0), dtype=torch.bool, device=images.device)
    else:
        mask = torch.rand(images.size(0), device=images.device) < p
    if not mask.any():
        return images
    flipped = torch.flip(images[mask], dims=[3])
    out = images.clone()
    out[mask] = flipped
    return out


def compute_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += param_norm.item() ** 2
    return total ** 0.5


def _build_inception_models(device: torch.device):
    weights = Inception_V3_Weights.IMAGENET1K_V1

    logits_model = inception_v3(
        weights=weights,
        transform_input=False,
        aux_logits=True,
    ).to(device)
    logits_model.eval()

    feats_model = inception_v3(
        weights=weights,
        transform_input=False,
        aux_logits=True,
    )
    feats_model.fc = nn.Identity()
    feats_model = feats_model.to(device)
    feats_model.eval()

    return logits_model, feats_model


def _get_inception_features_and_logits(
    images: torch.Tensor,
    logits_model: nn.Module,
    feats_model: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x + 1.0) * 0.5

        out_logits = logits_model(x)
        if isinstance(out_logits, torch.Tensor):
            logits = out_logits
        else:
            logits = out_logits.logits

        out_feats = feats_model(x)
        if isinstance(out_feats, torch.Tensor):
            feats = out_feats
        else:
            feats = out_feats.logits

    preds = F.softmax(logits, dim=1).cpu().numpy()
    feats_np = feats.cpu().numpy()
    return preds, feats_np


def _calculate_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)


def _calculate_inception_score(preds: np.ndarray, eps: float = 1e-16) -> float:
    p_yx = preds
    p_y = np.mean(p_yx, axis=0)
    kl = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    kl_sum = np.sum(kl, axis=1)
    is_score = np.exp(np.mean(kl_sum))
    return float(is_score)


def save_training_curves(
    history: Dict[str, List[float]],
    results_dir: str,
    scheduler_type: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    steps = np.arange(len(history["loss_D"]), dtype=np.int32)

    plt.figure()
    plt.plot(steps, history["loss_D"], label="loss_D")
    plt.plot(steps, history["loss_G"], label="loss_G")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"loss_curves_{scheduler_type}.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, history["noise_sigma"], label="noise_sigma")
    plt.plot(steps, history["augment_p"], label="augment_p")
    plt.plot(steps, history["reg_lambda"], label="reg_lambda")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"schedule_curves_{scheduler_type}.png"))
    plt.close()

    with open(
        os.path.join(results_dir, f"training_history_{scheduler_type}.json"),
        "w",
    ) as f:
        json.dump(history, f)


def sample_and_save_images(
    G: nn.Module,
    text_encoder: CLIPTextEncoder,
    val_loader: DataLoader,
    device: torch.device,
    results_dir: str,
    scheduler_type: str,
    noise_dim: int,
    image_size: int,
    n_samples: int = 64,
) -> None:
    G.eval()
    images_list: List[torch.Tensor] = []
    texts_list: List[str] = []

    with torch.no_grad():
        for images, texts in val_loader:
            images_list.append(images)
            texts_list.extend(list(texts))
            if sum(x.size(0) for x in images_list) >= n_samples:
                break

        images_cat = torch.cat(images_list, dim=0)[:n_samples]
        texts_sel = texts_list[:n_samples]
        texts_features = text_encoder(texts_sel).to(device)
        noise = torch.randn(n_samples, noise_dim, device=device)
        fake_images = G(noise, texts_features)
        fake_images = fake_images.cpu()

    os.makedirs(results_dir, exist_ok=True)
    grid = make_grid(fake_images, nrow=int(np.sqrt(n_samples)), normalize=True, value_range=(-1, 1))
    save_image(
        grid,
        os.path.join(results_dir, f"samples_grid_{scheduler_type}.png"),
    )


def evaluate_metrics(
    G: nn.Module,
    text_encoder: CLIPTextEncoder,
    val_loader: DataLoader,
    device: torch.device,
    n_eval_samples: int,
    noise_dim: int,
) -> Dict[str, float]:
    logits_model, feats_model = _build_inception_models(device)
    clip_model = text_encoder.model
    clip_model.eval()
    clip_preprocess = text_encoder.preprocess
    tokenizer = text_encoder.tokenizer

    G.eval()

    real_feats_list: List[np.ndarray] = []
    fake_feats_list: List[np.ndarray] = []
    is_preds_list: List[np.ndarray] = []
    clip_sims_list: List[float] = []

    to_pil = transforms.ToPILImage()

    processed = 0
    for images, texts in tqdm(val_loader, desc="Evaluating", leave=False):
        if processed >= n_eval_samples:
            break
        images = images.to(device)
        batch_size = images.size(0)
        remaining = n_eval_samples - processed
        if batch_size > remaining:
            images = images[:remaining]
            texts = list(texts)[:remaining]
            batch_size = images.size(0)

        with torch.no_grad():
            text_features = text_encoder(texts).to(device)
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = G(noise, text_features)

        preds_real, feats_real = _get_inception_features_and_logits(
            images, logits_model, feats_model, device
        )
        _, feats_fake = _get_inception_features_and_logits(
            fake_images, logits_model, feats_model, device
        )
        is_preds, _ = _get_inception_features_and_logits(
            fake_images, logits_model, feats_model, device
        )

        real_feats_list.append(feats_real)
        fake_feats_list.append(feats_fake)
        is_preds_list.append(is_preds)

        fake_for_clip = (fake_images.detach().cpu() + 1.0) * 0.5
        pil_images = [to_pil(img) for img in fake_for_clip]
        clip_imgs = torch.stack(
            [clip_preprocess(p) for p in pil_images],
            dim=0,
        ).to(device)
        with torch.no_grad():
            img_feats = clip_model.encode_image(clip_imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            tokens = tokenizer(list(texts)).to(device)
            txt_feats = clip_model.encode_text(tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            sims = torch.sum(img_feats * txt_feats, dim=-1)
        clip_sims_list.extend(sims.cpu().numpy().tolist())

        processed += batch_size

    real_feats = np.concatenate(real_feats_list, axis=0)
    fake_feats = np.concatenate(fake_feats_list, axis=0)
    is_preds_all = np.concatenate(is_preds_list, axis=0)
    clip_sims_arr = np.array(clip_sims_list, dtype=np.float32)

    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    mu_fake = np.mean(fake_feats, axis=0)
    sigma_fake = np.cov(fake_feats, rowvar=False)

    fid = _calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    inception_score = _calculate_inception_score(is_preds_all)
    clip_score = float(np.mean(clip_sims_arr))

    metrics = {
        "FID": fid,
        "InceptionScore": inception_score,
        "CLIPScore": clip_score,
        "n_eval_samples": int(processed),
    }
    return metrics


def train(
    data_root: str,
    scheduler_type: str = "annealed",
    image_size: int = 256,
    batch_size: int = 32,
    num_epochs: int = 20,
    device: str = "cuda",
    run_eval: bool = False,
    n_eval_samples: int = 5000,
    results_dir: str = "results",
    scheduler_kwargs: Dict[str, float] = None,
) -> Dict[str, float]:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    train_dataset = CocoTextImageDataset(
        root=data_root,
        split="train",
        transform=transform,
    )
    val_dataset = CocoTextImageDataset(
        root=data_root,
        split="val",
        transform=transform,
        max_samples=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=coco_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=coco_collate_fn,
    )

    text_encoder = CLIPTextEncoder(device=str(device_t))
    text_dim = text_encoder.text_dim

    G = ConditionalGenerator(
        noise_dim=128,
        text_dim=text_dim,
        base_channels=64,
        image_size=image_size,
    ).to(device_t)
    D = ConditionalDiscriminator(
        text_dim=text_dim,
        base_channels=64,
        image_size=image_size,
    ).to(device_t)

    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    if scheduler_type == "annealed":
        scheduler = AnnealedScheduler(**scheduler_kwargs)
        sched_params = []
    elif scheduler_type == "learnable":
        scheduler = LearnableMonotoneScheduler(**scheduler_kwargs)
        sched_params = list(scheduler.parameters())
    elif scheduler_type == "adaptive":
        scheduler = AdaptiveAnnealedScheduler(**scheduler_kwargs)
        sched_params = list(scheduler.parameters())
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    scheduler = scheduler.to(device_t)

    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_S = (
        torch.optim.Adam(sched_params, lr=1e-3)
        if len(sched_params) > 0
        else None
    )

    criterion = nn.BCEWithLogitsLoss()

    total_steps = num_epochs * len(train_loader)
    global_step = 0

    history: Dict[str, List[float]] = {
        "loss_D": [],
        "loss_G": [],
        "noise_sigma": [],
        "augment_p": [],
        "reg_lambda": [],
    }

    state_for_sched: Dict[str, float] = {
        "loss_g": 0.0,
        "loss_d": 0.0,
        "grad_norm_g": 0.0,
        "grad_norm_d": 0.0,
    }

    for epoch in range(num_epochs):
        G.train()
        D.train()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, texts in pbar:
            images = images.to(device_t)
            batch_size_curr = images.size(0)
            texts = list(texts)

            text_features = text_encoder(texts).to(device_t)

            sched_out = scheduler(global_step, total_steps, state_for_sched)
            noise_sigma = sched_out["noise_sigma"]
            aug_p = float(sched_out["augment_p"])
            reg_lambda = sched_out["reg_lambda"]

            valid = torch.ones(batch_size_curr, 1, device=device_t)
            fake = torch.zeros(batch_size_curr, 1, device=device_t)

            optim_D.zero_grad()
            real_images_aug = apply_augmentation(images, aug_p)
            real_logits = D(real_images_aug, text_features)
            loss_real = criterion(real_logits, valid)

            noise = torch.randn(batch_size_curr, 128, device=device_t) * noise_sigma
            fake_images = G(noise, text_features.detach())
            fake_images_aug = apply_augmentation(fake_images.detach(), aug_p)
            fake_logits = D(fake_images_aug, text_features)
            loss_fake = criterion(fake_logits, fake)

            loss_D = loss_real + loss_fake
            if reg_lambda.item() > 0.0:
                reg_loss = reg_lambda * (fake_images ** 2).mean()
                loss_D = loss_D + reg_loss

            loss_D.backward()
            grad_norm_d = compute_grad_norm(D.parameters())
            optim_D.step()

            optim_G.zero_grad()
            if optim_S is not None:
                optim_S.zero_grad()

            noise = torch.randn(batch_size_curr, 128, device=device_t) * noise_sigma
            fake_images = G(noise, text_features)
            fake_images_aug = apply_augmentation(fake_images, aug_p)
            logits_fake_for_g = D(fake_images_aug, text_features)
            loss_G = criterion(logits_fake_for_g, valid)

            loss_G.backward()
            grad_norm_g = compute_grad_norm(G.parameters())
            optim_G.step()
            if optim_S is not None:
                optim_S.step()

            epoch_loss_d += loss_D.item() * batch_size_curr
            epoch_loss_g += loss_G.item() * batch_size_curr

            history["loss_D"].append(float(loss_D.item()))
            history["loss_G"].append(float(loss_G.item()))
            history["noise_sigma"].append(float(noise_sigma.detach().cpu().item()))
            history["augment_p"].append(float(aug_p))
            history["reg_lambda"].append(float(reg_lambda.detach().cpu().item()))

            state_for_sched["loss_g"] = float(loss_G.item())
            state_for_sched["loss_d"] = float(loss_D.item())
            state_for_sched["grad_norm_g"] = float(grad_norm_g)
            state_for_sched["grad_norm_d"] = float(grad_norm_d)

            pbar.set_postfix(
                {
                    "loss_d": loss_D.item(),
                    "loss_g": loss_G.item(),
                    "noise": float(noise_sigma.detach().cpu().item()),
                    "aug_p": aug_p,
                    "reg": float(reg_lambda.detach().cpu().item()),
                }
            )

            global_step += 1

        epoch_loss_d /= len(train_dataset)
        epoch_loss_g /= len(train_dataset)
        print(
            f"Epoch {epoch+1}: "
            f"loss_D={epoch_loss_d:.4f}, "
            f"loss_G={epoch_loss_g:.4f}"
        )

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(ckpt_dir, f"gan_{scheduler_type}.pt"),
    )

    sched_results_dir = os.path.join(results_dir, scheduler_type)
    save_training_curves(history, sched_results_dir, scheduler_type)
    sample_and_save_images(
        G,
        text_encoder,
        val_loader,
        device_t,
        sched_results_dir,
        scheduler_type,
        noise_dim=128,
        image_size=image_size,
    )

    metrics: Dict[str, float] = {}
    if run_eval:
        metrics = evaluate_metrics(
            G,
            text_encoder,
            val_loader,
            device_t,
            n_eval_samples=n_eval_samples,
            noise_dim=128,
        )
        with open(
            os.path.join(
                sched_results_dir,
                f"metrics_{scheduler_type}.json",
            ),
            "w",
        ) as f:
            json.dump(metrics, f, indent=2)
        print(
            f"[{scheduler_type}] FID={metrics['FID']:.3f}, "
            f"IS={metrics['InceptionScore']:.3f}, "
            f"CLIPScore={metrics['CLIPScore']:.3f}"
        )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/COCO2017",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="annealed",
        choices=["annealed", "learnable", "adaptive"],
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=5000)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        scheduler_type=args.scheduler_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        run_eval=args.eval,
        n_eval_samples=args.eval_samples,
        results_dir=args.results_dir,
    )
