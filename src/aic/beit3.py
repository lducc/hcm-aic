import csv
import os
import sys
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import XLMRobertaTokenizer


@lru_cache
def load_beit3(device):
    sys.path.insert(0, os.environ["AIC_BEIT3_HOME"])
    import modeling_finetune
    import utils

    model = modeling_finetune.beit3_large_patch16_384_retrieval()
    utils.load_model_and_may_interpolate(
        os.environ["AIC_BEIT3_CHECKPOINT"], model, "model|module", ""
    )
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tokenizer = XLMRobertaTokenizer(os.environ["AIC_BEIT3_SPM"])
    return model, transform, tokenizer


def embed_beit3_images(paths, model, transform, device, batch_size=4):
    vectors = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            vision, _ = model(image=torch.stack(images).to(device), only_infer=True)
        vectors.append(vision.cpu().numpy().astype("float32"))
    return np.vstack(vectors)


def embed_beit3_text(queries, model, tokenizer, device):
    vectors = []
    for query in queries:
        tokens = tokenizer(
            query, max_length=64, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"].to(device)
        with torch.inference_mode():
            _, language = model(
                text_description=tokens,
                padding_mask=tokens.eq(tokenizer.pad_token_id),
                only_infer=True,
            )
        vectors.append(language.cpu().numpy().astype("float32"))
    return np.vstack(vectors)


def build_beit3_index(root):
    artifacts = root / "artifacts"
    with (artifacts / "frames.csv").open(newline="", encoding="utf-8") as file:
        frame_ids = {row["frame_uid"]: int(row["frame_id"]) for row in csv.DictReader(file)}
    index = None

    for feature_path in sorted((artifacts / "beit3").glob("*.npy")):
        video_id = feature_path.stem
        frame_dir = root / "keyframes" / "keyframes" / video_id
        if not frame_dir.exists():
            continue
        frames = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        features = np.load(feature_path, allow_pickle=False).astype("float32")
        if len(features) != len(frames):
            raise ValueError(f"{video_id}: feature/keyframe mismatch")
        if index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(features.shape[1]))
        ids = []
        for frame in frames:
            n = int(frame.stem)
            frame_uid = f"{video_id}__kf_{n}"
            if frame_uid not in frame_ids:
                raise ValueError(f"{frame_uid}: missing from frames.csv")
            ids.append(frame_ids[frame_uid])
        faiss.normalize_L2(features)
        index.add_with_ids(features, np.array(ids, dtype="int64"))

    artifacts.mkdir(exist_ok=True)
    if index is None:
        raise ValueError("no matching BEiT-3 features and keyframes found")
    faiss.write_index(index, str(artifacts / "beit3.faiss"))
    print(f"indexed {index.ntotal} BEiT-3 keyframes")


def search_beit3(root, queries, top_k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer = load_beit3(device)
    vectors = embed_beit3_text(queries, model, tokenizer, device)
    faiss.normalize_L2(vectors)
    index = faiss.read_index(str(root / "artifacts" / "beit3.faiss"))
    if index.d != vectors.shape[1]:
        raise ValueError("BEiT-3 index does not match the configured checkpoint")
    with (root / "artifacts" / "frames.csv").open(newline="", encoding="utf-8") as file:
        rows = {int(row["frame_id"]): row for row in csv.DictReader(file)}
    scores, ids = index.search(vectors, min(top_k, index.ntotal))
    best = {}
    for score_row, id_row in zip(scores, ids):
        for score, row_id in zip(score_row, id_row):
            if row_id not in best or score > best[row_id]:
                best[row_id] = float(score)
    results = []
    for rank, (row_id, score) in enumerate(sorted(best.items(), key=lambda item: item[1], reverse=True), 1):
        if rank > top_k:
            break
        result = dict(rows[row_id])
        result["timestamp_sec"] = float(result["timestamp_sec"])
        result["source"] = "beit3"
        result["raw_score"] = score
        result["rank"] = rank
        result["text"] = ""
        results.append(result)
    return results
