# AIC KIS Baseline

This project searches the HCMC AI Challenge 2025 video corpus. A text query returns
ranked keyframes with a video ID, timestamp, source modalities, and keyframe path.

## 1. Install Dependencies

Install the base environment for organizer CLIP search:

```bash
uv sync
```

Install every optional dependency for a full builder machine:

```bash
uv sync --all-extras
```

This installs Python packages only. It does not download model checkpoints,
organizer data, or shared search artifacts.

Install an optional extra only when using that feature:

```bash
uv sync --extra beit3  # BEiT-3 query encoder
uv sync --extra query  # Gemini query enhancement
uv sync --extra asr    # ChunkFormer transcription worker
uv sync --extra app    # Gradio search interface
```

`--extra asr` is not required to search an existing `asr.sqlite` database.
Teammates who only use the Gradio interface with shared artifacts need:

```bash
uv sync --extra app
```

## 2. Prepare Search Data

The repository does not store videos, keyframes, embeddings, indexes, or model
weights. Obtain them from organizer downloads, Colab, or the team's shared Drive folder.

There are two valid workflows.

### Data Builder

Use this workflow when you have extracted organizer CLIP features, keyframes, and
keyframe maps. Place them under:

```text
data/
  features/clip-features-32/
  keyframes/keyframes/
  map-keyframes/map-keyframes/
```

Build the organizer CLIP search artifacts:

```bash
uv run aic build
```

The command validates feature/keyframe counts, maps each keyframe to its organizer
timestamp, and writes:

```text
data/artifacts/clip.faiss
data/artifacts/frames.csv
```

`clip.faiss` stores CLIP vectors. `frames.csv` is the shared frame catalog: it maps
stable FAISS IDs to video IDs, timestamps, and keyframe paths. They must remain together.
When later ingestion creates new frames, it appends them to this catalog instead of creating
another model-specific CSV.

### Search User

Use this workflow when another team member has already built the artifacts. Copy the
shared files into:

```text
data/
  artifacts/
    clip.faiss
    frames.csv
    beit3.faiss        # optional
    asr.sqlite          # optional
  keyframes/keyframes/ # required to open returned images
```

Do not run `aic build` when `clip.faiss` and `frames.csv` are already supplied.

## 3. Run the CLI

Search with all available modalities:

```bash
uv run aic search "a person riding a bicycle"
```

Select one modality or an explicit combination for inspection:

```bash
uv run aic search "cảnh sát giao thông" --only asr
uv run aic search "a bicycle race" --only clip
uv run --env-file .env aic search "a bicycle race" --only clip --only beit3
```

CLI options:

| Option | Purpose |
| --- | --- |
| `--only clip` | Search the organizer CLIP index. |
| `--only beit3` | Search the BEiT-3 index. |
| `--only asr` | Search ChunkFormer transcripts. |
| `--top-k 20` | Return this many final results. |
| `--rerank` | Use nearby visual keyframes as additional support. |
| `--enhance` | Generate English visual query rewrites with Gemini. |

Without `--only`, search uses every artifact that is available. Repeating `--only`
selects a combination.

## 4. Configure Optional Models

### BEiT-3 Large 384

BEiT-3 requires its Python extra, an existing BEiT-3 index, and local model files.
Install the extra with `uv sync --extra beit3`, then download:

- [Microsoft UniLM BEiT-3 source](https://github.com/microsoft/unilm/tree/master/beit3)
- [BEiT-3 Large COCO retrieval checkpoint](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth)
- [BEiT-3 SentencePiece tokenizer](https://github.com/addf400/files/releases/download/beit3/beit3.spm)

Store them under `models/` and configure `.env`:

```dotenv
AIC_BEIT3_HOME=/absolute/path/to/AIC/models/beit3
AIC_BEIT3_CHECKPOINT=/absolute/path/to/AIC/models/beit3_large_patch16_384_coco_retrieval.pth
AIC_BEIT3_SPM=/absolute/path/to/AIC/models/beit3.spm
```

`AIC_BEIT3_HOME` is the UniLM `beit3` directory containing
`modeling_finetune.py` and `utils.py`. BEiT-3 is skipped automatically until the
Large 384 checkpoint and matching `beit3.faiss` are available.

### Gemini Query Enhancement

Install the query extra with `uv sync --extra query`. Add `GEMINI_API_KEY` to `.env`:

```dotenv
GEMINI_API_KEY=...
```

Run enhanced visual search:

```bash
uv run --env-file .env aic search "Tìm cảnh đua xe đạp từ trên cao" --enhance
```

The command prints three English visual rewrites for CLIP and BEiT-3. ASR always
receives the original query.

## 5. Run the Search UI

Install the app extra, then start the local Gradio interface:

```bash
uv sync --extra app
uv run aic app
```

The interface reuses the CLI search function. It keeps loaded visual models in
memory while it runs, shows keyframes when local paths are available, and lists
the video ID, timestamp, modalities, and fused score for every result.

## 6. Run Colab Workers

Use separate GPU Colab runtimes for visual embeddings and ASR. Both notebooks mount
Drive, download organizer archives to `/content/work`, process one archive at a time,
save resumable per-video output to Drive, then delete temporary archive data.

### BEiT-3 Visual Worker

Open [ingest_beit3.ipynb](notebooks/ingest_beit3.ipynb) in Colab and run all cells.
It replaces the previous Base vectors once, then processes the listed L21-L30 keyframe
archive shards and writes:

```text
AIC_ARTIFACTS/beit3/L23_V001.npy
...
AIC_ARTIFACTS/beit3.faiss
AIC_ARTIFACTS/frames.csv
```

Existing per-video `.npy` files are skipped after a restart.
`model.txt` in the `beit3` directory records the active model so resumed workers do not
mix Base and Large vectors.

### ChunkFormer ASR Worker

Open [ingest_chunkformer.ipynb](notebooks/ingest_chunkformer.ipynb) in Colab and run
all cells. It processes the listed L21-L30 video archive shards and writes:

```text
AIC_ARTIFACTS/asr/L23_V001.json
...
AIC_ARTIFACTS/asr.sqlite
```

Existing per-video transcript JSON files are skipped after a restart. ChunkFormer is
licensed CC-BY-NC-4.0.

Copy `beit3.faiss`, `frames.csv`, and `asr.sqlite` from Drive into local
`data/artifacts/` when they are ready.

## Retrieval Architecture

```text
text query
  |- CLIP ViT-B/32 -> clip.faiss
  |- BEiT-3 Large  -> beit3.faiss
  `- ASR keywords  -> asr.sqlite
                     |
                     `- frames.csv + reciprocal-rank fusion -> ranked keyframes
```

CLIP and BEiT-3 use separate embedding spaces and indexes, but both return stable IDs
from `frames.csv`. This lets an index cover a different subset of frames without a
second metadata file. ASR hits are snapped to the nearest organizer keyframe before fusion.
