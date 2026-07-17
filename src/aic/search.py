import csv
import os
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch


@lru_cache
def load_clip(device):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    return model, tokenizer


def clip_results(root, queries, top_k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_clip(device)
    with torch.inference_mode():
        vectors = model.encode_text(tokenizer(queries).to(device)).cpu().numpy().astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.read_index(str(root / "artifacts" / "clip.faiss"))
    with (root / "artifacts" / "frames.csv").open(newline="", encoding="utf-8") as file:
        rows = {int(row["frame_id"]): row for row in csv.DictReader(file)}
    scores, ids = index.search(vectors, min(top_k, index.ntotal))
    best = {}
    for score_row, id_row in zip(scores, ids):
        for score, row_id in zip(score_row, id_row):
            if row_id not in best or score > best[row_id]:
                best[row_id] = float(score)
    results = []
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    for rank, (row_id, score) in enumerate(ranked[:top_k], 1):
        result = dict(rows[row_id])
        result["timestamp_sec"] = float(result["timestamp_sec"])
        result["source"] = "clip"
        result["raw_score"] = score
        result["rank"] = rank
        result["text"] = ""
        results.append(result)
    return results


def merge_results(lists, candidate_k):
    buckets = {}
    for results in lists:
        for result in results:
            key = (result["video_id"], int(float(result["timestamp_sec"]) / 5))
            if key not in buckets:
                buckets[key] = {
                    "video_id": result["video_id"],
                    "timestamp_sec": result["timestamp_sec"],
                    "frame_path": result["frame_path"],
                    "text": result["text"],
                    "sources": {},
                }
            item = buckets[key]
            old = item["sources"].get(result["source"])
            if old is None or result["rank"] < old["rank"]:
                item["sources"][result["source"]] = {
                    "rank": result["rank"],
                    "score": result["raw_score"],
                    "weight": result.get("weight", 1),
                }
            if not item["frame_path"] and result["frame_path"]:
                item["frame_path"] = result["frame_path"]
            if result["text"]:
                item["text"] = result["text"]

    results = []
    for result in buckets.values():
        result["rrf_score"] = 0
        for source in result["sources"].values():
            result["rrf_score"] += source["weight"] / (60 + source["rank"])
        result["neighbor_score"] = 0
        result["final_score"] = result["rrf_score"]
        results.append(result)
    results.sort(key=lambda item: item["rrf_score"], reverse=True)
    results = results[:candidate_k]
    for rank, result in enumerate(results, 1):
        result["rrf_rank"] = rank
    return results


def rerank_neighbors(results):
    for result in results[:50]:
        if "clip" not in result["sources"] and "beit3" not in result["sources"]:
            continue
        before = []
        after = []
        for other in results:
            if other is result or other["video_id"] != result["video_id"]:
                continue
            if "clip" not in other["sources"] and "beit3" not in other["sources"]:
                continue
            distance = other["timestamp_sec"] - result["timestamp_sec"]
            if distance < -15 or distance > 15:
                continue
            if distance < 0:
                before.append(other)
            elif distance > 0:
                after.append(other)
        before.sort(key=lambda item: item["timestamp_sec"], reverse=True)
        after.sort(key=lambda item: item["timestamp_sec"])
        support = 0
        for other in before[:2] + after[:2]:
            support += other["rrf_score"] / (1 + abs(other["timestamp_sec"] - result["timestamp_sec"]) / 5)
        result["neighbor_score"] = support
        result["final_score"] += 0.5 * support
    return results


def deduplicate(results, top_k):
    results.sort(key=lambda item: item["final_score"], reverse=True)
    kept = []
    for result in results:
        duplicate = False
        for other in kept:
            if result["video_id"] == other["video_id"] and abs(result["timestamp_sec"] - other["timestamp_sec"]) <= 5:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(result)
        if len(kept) == top_k:
            break
    for rank, result in enumerate(kept, 1):
        result["rank"] = rank
    return kept


def snap_asr_results(root, results):
    metadata_path = root / "artifacts" / "frames.csv"
    if not metadata_path.exists():
        return results
    frames_by_video = {}
    with metadata_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frames_by_video.setdefault(row["video_id"], []).append(row)
    for result in results:
        frames = frames_by_video.get(result["video_id"], [])
        if not frames:
            continue
        nearest = min(
            frames,
            key=lambda frame: abs(float(frame["timestamp_sec"]) - result["timestamp_sec"]),
        )
        result["frame_path"] = nearest["frame_path"]
    return results


def search(root, query, top_k, rerank=False, enhance=False, visual_queries=None, modalities=None):
    if visual_queries is None:
        visual_queries = [query]
    if enhance:
        from aic.query import enhance as rewrite

        visual_queries = rewrite(query)
    candidate_k = max(50, 5 * top_k)
    lists = []
    selected = modalities or ["clip", "beit3", "asr"]
    if "clip" in selected and (root / "artifacts" / "clip.faiss").exists() and (root / "artifacts" / "frames.csv").exists():
        lists.append(clip_results(root, visual_queries, candidate_k))
    beit3_index = root / "artifacts" / "beit3.faiss"
    checkpoint = Path(os.environ.get("AIC_BEIT3_CHECKPOINT", ""))
    has_beit3_runtime = all(
        os.environ.get(name)
        for name in ["AIC_BEIT3_HOME", "AIC_BEIT3_CHECKPOINT", "AIC_BEIT3_SPM"]
    )
    beit3_ready = (
        has_beit3_runtime
        and checkpoint.name == "beit3_large_patch16_384_coco_retrieval.pth"
        and beit3_index.exists()
        and faiss.read_index(str(beit3_index)).d == 1024
    )
    if "beit3" in selected and beit3_ready and (root / "artifacts" / "frames.csv").exists():
        from aic.beit3 import search_beit3

        lists.append(search_beit3(root, visual_queries, candidate_k))
    if "asr" in selected and (root / "artifacts" / "asr.sqlite").exists():
        from aic.asr import search_asr

        lists.append(snap_asr_results(root, search_asr(root, query, candidate_k)))
    results = merge_results(lists, candidate_k)
    if rerank:
        rerank_neighbors(results)
    return deduplicate(results, top_k)
