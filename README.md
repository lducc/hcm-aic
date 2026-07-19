# AIC KIS Baseline

Local multimodal search for the HCMC AI Challenge 2025 video corpus. A Vietnamese or
English query returns ranked keyframes with a video ID, timestamp, available evidence,
and a local frame path.

```text
query
  |- OpenAI CLIP ViT-B/32 -> clip.faiss
  |- BEiT-3 Large 384    -> beit3.faiss
  `- ChunkFormer ASR     -> asr.sqlite
                                |
                                `- reciprocal-rank fusion -> frames.csv -> results
```

`frames.csv` is the common catalog. It maps FAISS IDs to keyframes, video IDs, and
timestamps, so visual and ASR results describe the same searchable moments.

## Install Dependencies

Install the complete student environment. This includes local search, BEiT-3 runtime,
ChunkFormer ingestion, Gemini query enhancement, and Gradio. It does not download data,
model checkpoints, or generated artifacts.

```bash
uv sync
```

## Prepare Data

Set up one data root. The default is `data`; set `AIC_DATA` only when using a second
bundle such as `data/l23`.

### Shared Drive Data

- [Organizer AIC 2025 data](https://drive.google.com/drive/folders/1eO4XpkeF0gq1J5P5-_N4TUMqQ_c9vn4R?usp=drive_link) contains the original ZIP archives: videos, keyframes, organizer CLIP vectors, and keyframe maps.
- [Team processed artifacts](https://drive.google.com/drive/folders/1eD3UOK5QPu9mKe6Yabj9RHZNavcWsm7j?usp=sharing) contains resumable worker outputs and finished releases.

The organizer archive is the source material. The processed-artifact Drive is what most
teammates use for search. Do not download the complete organizer dataset just to run the
app.

```text
data/
  artifacts/
    clip.faiss          # organizer CLIP index
    beit3.faiss         # optional BEiT-3 index
    asr.sqlite          # optional ASR index
    frames.csv          # required catalog for both visual indexes
  keyframes/keyframes/  # required to show result images
```

Use matching files from one shared release. Do not combine an L23 index with a full
Batch 1 `frames.csv`.

### L23 Example Bundle

L23 is the small end-to-end example for local development. From the processed-artifact
Drive, copy the matching `l23-v1` release files and L23 keyframes into:

```text
data/l23/
  artifacts/
    clip.faiss
    beit3.faiss
    asr.sqlite
    frames.csv
  keyframes/keyframes/L23_V001/
  keyframes/keyframes/L23_V002/
  ...
```

Source videos are optional for search. To inspect or play L23 source video later,
download `Videos_L23_a.zip` from the organizer Drive and extract it under:

```text
data/l23/videos/L23_V001.mp4
data/l23/videos/L23_V002.mp4
...
```

Run the example bundle without affecting the default data root:

```bash
AIC_DATA=data/l23 uv run --env-file .env aic search "xe đạp" --top-k 10
```

Do not download `workers/beit3/*.npy` or `workers/asr/*.json` for normal search. They
exist only so Colab ingestion can resume after a disconnect.

### Build Organizer CLIP

Run this only on a machine with extracted organizer features, keyframes, and mappings:

```text
data/
  features/clip-features-32/
  keyframes/keyframes/
  map-keyframes/map-keyframes/
```

```bash
uv run aic build
```

The command validates feature/keyframe counts, maps keyframe numbers to timestamps, and
writes `artifacts/clip.faiss` and `artifacts/frames.csv`.

For normal team use, download the prepared release instead. Raw videos, worker `.npy`
files, and model weights are unnecessary unless that teammate is ingesting data.

## Configure Optional Models

### BEiT-3 Large

Download the [UniLM BEiT-3 source](https://github.com/microsoft/unilm/tree/master/beit3),
the [Large COCO retrieval checkpoint](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth),
and the [SentencePiece tokenizer](https://github.com/addf400/files/releases/download/beit3/beit3.spm).

Place them under `models/`, then copy `.env.example` to `.env` and set:

```dotenv
AIC_BEIT3_HOME=/absolute/path/to/AIC/models/beit3
AIC_BEIT3_CHECKPOINT=/absolute/path/to/AIC/models/beit3_large_patch16_384_coco_retrieval.pth
AIC_BEIT3_SPM=/absolute/path/to/AIC/models/beit3.spm
```

BEiT-3 activates automatically only when these three values and a compatible
`beit3.faiss` are available. Otherwise search continues with CLIP and ASR.

### Gemini Query Enhancement

Set `GEMINI_API_KEY` in `.env` to enable `--enhance`. It prints three English visual
rewrites for CLIP and BEiT-3; ASR keeps the original query.

## Search

```bash
uv run aic search "a person riding a bicycle"
uv run aic search "cảnh sát giao thông" --only asr
uv run --env-file .env aic search "a bicycle race" --only clip --only beit3 --rerank
uv run --env-file .env aic search "Tìm cảnh đua xe đạp từ trên cao" --enhance
```

`--only` may be repeated to compare a single modality or a combination. `--rerank`
adds support from nearby visual keyframes.

## Run the App

```bash
uv run --env-file .env aic app
```

The Gradio app uses the same search function as the CLI and keeps models loaded while
the process remains running.

## Colab

The notebooks are standalone and use the shared Drive layout below. They do not clone
this repository.

```text
AIC_ARTIFACTS/
  workers/beit3/        # resumable BEiT-3 .npy outputs
  workers/asr/          # resumable ChunkFormer JSON outputs
  releases/l23-v1/      # L23 BEiT-3 index and frames.csv
  releases/full-v1/     # full Batch 1 indexes and frames.csv
```

- `ingest_beit3.ipynb` processes L23 keyframes and publishes `l23-v1/beit3.faiss` and
  `l23-v1/frames.csv`.
- `ingest_chunkformer.ipynb` reads `MyDrive/AIC2025/video_batch_1`, saves resumable
  transcripts, and publishes `full-v1/asr.sqlite` after every Batch 1 video is done.

Copy a release into a local data root using the structure in **Prepare Data**. The
ChunkFormer model is licensed CC-BY-NC-4.0; confirm that it is acceptable for your use.
