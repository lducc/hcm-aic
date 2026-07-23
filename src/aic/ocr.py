import csv
import json
import os
import re
import sqlite3
from functools import lru_cache

COMMON_WORDS = {
    "a", "an", "the", "các", "có", "của", "đang", "được", "là", "một",
    "người", "những", "ở", "trong", "và", "với",
}


@lru_cache
def load_paddleocr(lang):
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def ocr_paddle(image_path, lang="vi", min_confidence=0.5):
    engine = load_paddleocr(lang)
    pages = engine.ocr(str(image_path), cls=True)
    lines = []
    for page in pages or []:
        for _, (text, confidence) in page or []:
            if confidence >= min_confidence:
                lines.append(text)
    return " ".join(lines)


def ocr_tesseract(image_path, lang="vie+eng"):
    import pytesseract
    from PIL import Image

    with Image.open(image_path) as image:
        return pytesseract.image_to_string(image, lang=lang).strip()


def ocr_frame(image_path):
    engine = os.environ.get("AIC_OCR_ENGINE", "paddleocr")
    if engine == "tesseract":
        return ocr_tesseract(image_path, os.environ.get("AIC_OCR_LANG", "vie+eng"))
    if engine == "paddleocr":
        return ocr_paddle(image_path, os.environ.get("AIC_OCR_LANG", "vi"))
    raise ValueError(f"unknown AIC_OCR_ENGINE: {engine}")


def transcribe_video(frame_dir, output_path):
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    results = []
    for frame in frames:
        text = ocr_frame(frame)
        if text:
            results.append({"keyframe_n": int(frame.stem), "text": text})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)


def build_ocr_index(root):
    (root / "ocr").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    with (root / "artifacts" / "frames.csv").open(newline="", encoding="utf-8") as file:
        frames_by_uid = {row["frame_uid"]: row for row in csv.DictReader(file)}
    with sqlite3.connect(root / "artifacts" / "ocr.sqlite") as connection:
        connection.execute("DROP TABLE IF EXISTS ocr")
        connection.execute(
            "CREATE VIRTUAL TABLE ocr USING fts5(video_id UNINDEXED, timestamp_sec UNINDEXED, frame_path UNINDEXED, text)"
        )
        for path in sorted((root / "ocr").glob("*.json")):
            video_id = path.stem
            for item in json.loads(path.read_text(encoding="utf-8")):
                frame = frames_by_uid.get(f"{video_id}__kf_{item['keyframe_n']}")
                if frame is None:
                    continue
                connection.execute(
                    "INSERT INTO ocr(video_id, timestamp_sec, frame_path, text) VALUES (?, ?, ?, ?)",
                    (video_id, float(frame["timestamp_sec"]), frame["frame_path"], item["text"]),
                )


def search_ocr(root, query, top_k):
    words = re.findall(r"\w+", query, flags=re.UNICODE)
    if not words:
        return []
    phrases = [f'"{" ".join(words)}"']
    for first, second in zip(words, words[1:]):
        if first not in COMMON_WORDS and second not in COMMON_WORDS:
            phrases.append(f'"{first} {second}"')
    if len(phrases) == 1:
        for word in words:
            if word not in COMMON_WORDS:
                phrases.append(f'"{word}"')
    with sqlite3.connect(root / "artifacts" / "ocr.sqlite") as connection:
        rows = connection.execute(
            "SELECT video_id, timestamp_sec, frame_path, text, bm25(ocr) FROM ocr WHERE ocr MATCH ? ORDER BY bm25(ocr) LIMIT ?",
            (" OR ".join(phrases), top_k),
        ).fetchall()
    results = []
    phrase = " ".join(words).casefold()
    for rank, (video_id, timestamp_sec, frame_path, text, score) in enumerate(rows, 1):
        weight = 3 if phrase in text.casefold() else 1
        results.append({
            "video_id": video_id,
            "timestamp_sec": timestamp_sec,
            "frame_path": frame_path,
            "text": text,
            "source": "ocr",
            "raw_score": score,
            "rank": rank,
            "weight": weight,
        })
    return results
