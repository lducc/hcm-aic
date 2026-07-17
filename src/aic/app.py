import os
from pathlib import Path

import gradio as gr

from aic.search import search


def make_app(root):
    def run_search(query, top_k, rerank):
        if not query.strip():
            return [], []

        results = search(root, query, int(top_k), rerank)
        gallery = []
        table = []
        for result in results:
            sources = ", ".join(result["sources"])
            caption = (
                f"#{result['rank']}  {result['video_id']}  "
                f"{float(result['timestamp_sec']):.2f}s  {sources}"
            )
            frame_path = root / result["frame_path"]
            if result["frame_path"] and frame_path.exists():
                gallery.append((str(frame_path), caption))
            table.append(
                [
                    result["rank"],
                    result["video_id"],
                    round(float(result["timestamp_sec"]), 2),
                    sources,
                    round(result["final_score"], 4),
                    result["frame_path"] or "-",
                ]
            )
        return gallery, table

    with gr.Blocks(title="AIC KIS Search") as app:
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="Describe a video moment")
            top_k = gr.Slider(5, 50, value=20, step=1, label="Results")
            rerank = gr.Checkbox(label="Use nearby keyframes", value=False)
        button = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(label="Keyframes", columns=4, height="auto")
        table = gr.Dataframe(
            headers=["Rank", "Video", "Time", "Sources", "Fused", "Frame path"],
            interactive=False,
        )
        button.click(run_search, [query, top_k, rerank], [gallery, table])
        query.submit(run_search, [query, top_k, rerank], [gallery, table])
    return app


def main(root=None):
    root = root or Path(os.environ.get("AIC_DATA", "data"))
    make_app(root).launch()
