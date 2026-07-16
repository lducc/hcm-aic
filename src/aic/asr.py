import json
import re
import sqlite3


def transcribe_video(model, video_path, output_path):
    result = model.endless_decode(
        audio_path=str(video_path),
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=14400,
        return_timestamps=True,
    )
    segments = []
    for item in result:
        start_hours, start_minutes, start_seconds, start_milliseconds = map(
            int, item["start"].split(":")
        )
        end_hours, end_minutes, end_seconds, end_milliseconds = map(
            int, item["end"].split(":")
        )
        segments.append(
            {
                "start": start_hours * 3600 + start_minutes * 60 + start_seconds + start_milliseconds / 1000,
                "end": end_hours * 3600 + end_minutes * 60 + end_seconds + end_milliseconds / 1000,
                "text": item["decode"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(segments, file, ensure_ascii=False)


def build_asr_index(root):
    (root / "asr").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "artifacts" / "asr.sqlite") as connection:
        connection.execute("DROP TABLE IF EXISTS asr")
        connection.execute("CREATE VIRTUAL TABLE asr USING fts5(video_id UNINDEXED, start_sec UNINDEXED, end_sec UNINDEXED, text)")
        for path in sorted((root / "asr").glob("*.json")):
            video_id = path.stem
            for segment in json.loads(path.read_text(encoding="utf-8")):
                connection.execute(
                    "INSERT INTO asr(video_id, start_sec, end_sec, text) VALUES (?, ?, ?, ?)",
                    (video_id, segment["start"], segment["end"], segment["text"]),
                )


def search_asr(root, query, top_k):
    words = re.findall(r"\w+", query, flags=re.UNICODE)
    if not words:
        return []
    phrases = [f'"{" ".join(words)}"']
    common_words = {
        "a", "an", "the", "các", "có", "của", "đang", "được", "là", "một",
        "người", "những", "ở", "trong", "và", "với",
    }
    for first, second in zip(words, words[1:]):
        if first not in common_words and second not in common_words:
            phrases.append(f'"{first} {second}"')
    if len(phrases) == 1:
        for word in words:
            if word not in common_words:
                phrases.append(f'"{word}"')
    with sqlite3.connect(root / "artifacts" / "asr.sqlite") as connection:
        rows = connection.execute(
            "SELECT video_id, start_sec, end_sec, text, bm25(asr) FROM asr WHERE asr MATCH ? ORDER BY bm25(asr) LIMIT ?",
            (" OR ".join(phrases), top_k),
        ).fetchall()
    results = []
    phrase = " ".join(words).casefold()
    for rank, (video_id, start, end, text, score) in enumerate(rows, 1):
        weight = 3 if phrase in text.casefold() else 1
        results.append({
            "video_id": video_id,
            "timestamp_sec": (start + end) / 2,
            "frame_path": "",
            "text": text,
            "source": "asr",
            "raw_score": score,
            "rank": rank,
            "weight": weight,
        })
    return results
