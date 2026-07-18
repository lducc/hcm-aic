import csv

import faiss
import numpy as np


def build(root):
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    rows = []
    frame_ids = {}
    catalog_path = artifacts / "frames.csv"
    if catalog_path.exists():
        with catalog_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            frame_ids[row["frame_uid"]] = int(row["frame_id"])
    next_frame_id = max(frame_ids.values(), default=-1) + 1
    index = None
    indexed = 0

    for feature_file in sorted((root / "features" / "clip-features-32").glob("*.npy")):
        video_id = feature_file.stem
        frame_dir = root / "keyframes" / "keyframes" / video_id
        if not frame_dir.exists():
            continue

        frames = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        features = np.load(feature_file, allow_pickle=False).astype("float32")
        if len(features) != len(frames):
            raise ValueError(f"{video_id}: feature/keyframe mismatch")

        faiss.normalize_L2(features)
        if index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(features.shape[1]))

        with (root / "map-keyframes" / "map-keyframes" / f"{video_id}.csv").open() as file:
            times = {
                int(row["n"]): float(row["pts_time"])
                for row in csv.DictReader(file)
            }

        ids = []
        for frame in frames:
            n = int(frame.stem)
            frame_uid = f"{video_id}__kf_{n}"
            if frame_uid not in frame_ids:
                frame_ids[frame_uid] = next_frame_id
                rows.append({
                    "frame_id": next_frame_id,
                    "frame_uid": frame_uid,
                    "video_id": video_id,
                    "keyframe_n": n,
                    "timestamp_sec": times[n],
                    "frame_path": str(frame.relative_to(root)),
                    "source": "organizer",
                })
                next_frame_id += 1
            ids.append(frame_ids[frame_uid])
        index.add_with_ids(features, np.array(ids, dtype="int64"))
        indexed += len(features)

    if index is None:
        raise ValueError("no matching organizer features and keyframes found")
    faiss.write_index(index, str(artifacts / "clip.faiss"))

    with catalog_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    assert indexed == index.ntotal
    print(f"indexed {index.ntotal} keyframes")
