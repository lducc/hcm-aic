import argparse
import os
from pathlib import Path

from aic.build import build
from aic.search import search


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    commands.add_parser("app")
    search_parser = commands.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=20)
    search_parser.add_argument("--rerank", action="store_true")
    search_parser.add_argument("--enhance", action="store_true")
    search_parser.add_argument("--only", choices=["clip", "beit3", "asr", "ocr"], action="append")
    args = parser.parse_args()
    root = Path(os.environ.get("AIC_DATA", "data"))

    if args.command == "build":
        build(root)
        return

    if args.command == "app":
        from aic.app import main as run_app

        run_app(root)
        return

    visual_queries = None
    if args.enhance:
        from aic.query import enhance

        visual_queries = enhance(args.query)
        print("enhanced visual queries:")
        for prompt in visual_queries:
            print(f"- {prompt}")

    print("rank  fused  video_id   time     sources             frame_path")
    for result in search(
        root,
        args.query,
        args.top_k,
        args.rerank,
        visual_queries=visual_queries,
        modalities=args.only,
    ):
        print(
            f"{result['rank']:<4}  {result['final_score']:.3f}  {result['video_id']:<10} "
            f"{float(result['timestamp_sec']):<7.2f}  {','.join(result['sources']):<18}  "
            f"{result['frame_path'] or '-'}"
        )
