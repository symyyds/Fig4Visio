#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_blank_scene(args: argparse.Namespace) -> dict:
    assets = []
    if args.image:
        assets.append(
            {
                "id": "source-image",
                "kind": "source_image",
                "path": str(Path(args.image).resolve()),
            }
        )
    if args.style_ref:
        assets.append(
            {
                "id": "style-reference",
                "kind": "style_reference",
                "path": str(Path(args.style_ref).resolve()),
            }
        )

    return {
        "version": "0.1",
        "metadata": {
            "title": args.title,
            "created_by": "visiomaster.image_to_scene",
            "style_profile": args.style_profile,
            "source_image": str(Path(args.image).resolve()) if args.image else None,
            "style_reference": str(Path(args.style_ref).resolve()) if args.style_ref else None,
            "notes": [
                "Starter scene only. Replace nodes and edges after visual analysis.",
                "Coordinates use top-left page origin and inches.",
                "Prefer editable reconstruction over full-image embedding."
            ],
        },
        "page": {
            "width": args.page_width,
            "height": args.page_height,
            "units": "in",
            "origin": "top-left",
            "background": "#FFFFFF",
        },
        "nodes": [],
        "edges": [],
        "assets": assets,
    }


def load_template(template_name: str) -> dict:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "templates"
        / "examples"
        / f"{template_name}.scene.json"
    )
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return json.loads(template_path.read_text(encoding="utf-8"))


def merge_source_metadata(scene: dict, args: argparse.Namespace) -> dict:
    metadata = scene.setdefault("metadata", {})
    assets = scene.setdefault("assets", [])

    metadata["title"] = args.title or metadata.get("title") or "VisioMaster Scene"
    metadata["created_by"] = "visiomaster.image_to_scene"
    metadata["style_profile"] = args.style_profile or metadata.get("style_profile") or "paper_white"

    if args.image:
        image_path = str(Path(args.image).resolve())
        metadata["source_image"] = image_path
        assets.append({"id": "source-image", "kind": "source_image", "path": image_path})

    if args.style_ref:
        style_path = str(Path(args.style_ref).resolve())
        metadata["style_reference"] = style_path
        assets.append({"id": "style-reference", "kind": "style_reference", "path": style_path})

    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a starter scene.json for visiomaster.",
    )
    parser.add_argument("--image", help="Optional source image path.")
    parser.add_argument("--style-ref", help="Optional style reference image path.")
    parser.add_argument(
        "--template",
        choices=["blank", "basic-flow"],
        default="blank",
        help="Starter template.",
    )
    parser.add_argument("--title", default="VisioMaster Scene")
    parser.add_argument("--page-width", type=float, default=13.333)
    parser.add_argument("--page-height", type=float, default=7.5)
    parser.add_argument(
        "--style-profile",
        choices=["paper_white", "clean_white"],
        default="paper_white",
    )
    parser.add_argument("--output", required=True, help="Output scene.json path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    if args.template == "blank":
        scene = build_blank_scene(args)
    else:
        scene = merge_source_metadata(load_template("basic_flow"), args)
        scene["page"]["width"] = args.page_width
        scene["page"]["height"] = args.page_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote starter scene: {output_path}")
    print("Next steps:")
    print("1. Edit nodes, edges, and styles.")
    print("2. Validate with scene_validate.py.")
    print("3. Render with scene_to_visio.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
