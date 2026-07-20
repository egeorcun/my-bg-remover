"""bgr CLI: uv run bgr remove input.jpg -o output.png"""
import argparse

from PIL import Image

from bgr.pipeline import PipelineSegmenter
from bgr.registry import get_segmenter


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bgr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rm = sub.add_parser("remove", help="remove the background")
    rm.add_argument("input")
    rm.add_argument("-o", "--output", required=True)
    rm.add_argument("--model", default="rmbg-2.0")
    rm.add_argument("--refine", action="store_true")
    rm.add_argument("--no-decontaminate", action="store_true")
    a = ap.parse_args(argv)

    seg = get_segmenter(a.model)
    pipe = seg if isinstance(seg, PipelineSegmenter) else PipelineSegmenter(seg)
    if a.refine and not pipe.refine:
        pipe = PipelineSegmenter(pipe.base, refine=True)
    out = pipe.process(Image.open(a.input), decontaminate=not a.no_decontaminate)
    out.save(a.output)
    print(f"saved: {a.output}")


if __name__ == "__main__":
    main()
