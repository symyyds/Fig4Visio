#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import gui_app  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def default_material_dir() -> Path:
    return Path.home() / "Desktop" / "\u79d1\u7814\u7ed8\u56fe" / "\u9876\u4f1a\u9876\u520a\u7d20\u6750"


def image_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_counts(scene_path: Path) -> dict[str, int]:
    scene = load_json(scene_path)
    nodes = scene.get("nodes", [])
    edges = scene.get("edges", [])
    assets = scene.get("assets", [])
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "assets": len(assets),
        "image_tiles": sum(1 for node in nodes if node.get("type") == "image_tile"),
        "icon_regions": int(scene.get("metadata", {}).get("icon_vector_regions") or 0),
        "icon_parts": int(scene.get("metadata", {}).get("icon_vector_parts") or 0),
        "icon_strokes": sum(1 for edge in edges if edge.get("semantic_role") == "editable_icon_stroke"),
        "icon_polygons": sum(1 for node in nodes if node.get("semantic_role") == "editable_icon_polygon"),
    }


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def make_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    passed_rows = [row for row in rows if row.get("self_check_png") and Path(row["self_check_png"]).exists()]
    if not passed_rows:
        return

    thumb_w = 680
    thumb_h = 150
    label_h = 34
    cols = 1
    rows_count = len(passed_rows)
    sheet = Image.new("RGB", (thumb_w * cols, (thumb_h + label_h) * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, row in enumerate(passed_rows):
        y = index * (thumb_h + label_h)
        label = (
            f"{row['index']:02d}. {row['name']} | "
            f"status={row.get('quality_status')} | score={row.get('self_check_score'):.3f} | "
            f"no_embed={row.get('no_image_embedding')} | mode={row.get('selected_mode')}"
        )
        draw.text((8, y + 8), label, fill="#111111", font=font)
        image = Image.open(row["self_check_png"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (0, y + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def summarize(rows: list[dict[str, Any]], batch_dir: Path, source_dir: Path) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    return {
        "source_dir": str(source_dir),
        "batch_dir": str(batch_dir.resolve()),
        "count": len(rows),
        "ok_count": len(ok_rows),
        "failed_count": len(rows) - len(ok_rows),
        "download_allowed_count": sum(1 for row in ok_rows if row.get("download_allowed")),
        "self_check_passed_count": sum(1 for row in ok_rows if row.get("self_check_passed")),
        "no_image_embedding_count": sum(1 for row in ok_rows if row.get("no_image_embedding")),
        "vsdx_no_media_count": sum(1 for row in ok_rows if not row.get("vsdx_has_embedded_images")),
        "scene_assets_total": sum(int(row.get("assets") or 0) for row in ok_rows),
        "scene_image_tiles_total": sum(int(row.get("image_tiles") or 0) for row in ok_rows),
        "vsdx_media_count_total": sum(int(row.get("vsdx_media_count") or 0) for row in ok_rows),
        "icon_regions_total": sum(int(row.get("icon_regions") or 0) for row in ok_rows),
        "icon_parts_total": sum(int(row.get("icon_parts") or 0) for row in ok_rows),
        "min_self_check_score": min((float(row.get("self_check_score") or 0.0) for row in ok_rows), default=0.0),
        "rows": rows,
    }


def run_batch(source_dir: Path, batch_dir: Path) -> int:
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "batch_workflow_check.log"
    rows: list[dict[str, Any]] = []
    files = image_files(source_dir)
    append_log(log_path, f"source_dir={source_dir}")
    append_log(log_path, f"image_count={len(files)}")

    for index, image_path in enumerate(files, 1):
        row: dict[str, Any] = {"index": index, "name": image_path.name, "source": str(image_path)}
        append_log(log_path, f"START {index}/{len(files)} {image_path.name}")
        item_logs: list[str] = []

        def item_log(message: str) -> None:
            item_logs.append(str(message))
            if "score=" in str(message) or "self" in str(message).lower() or "\u81ea\u68c0" in str(message):
                append_log(log_path, f"{index:02d} {message}")

        try:
            output = gui_app.run_visiomaster_job(image_path, log=item_log)
            counts = scene_counts(output.scene)
            vsdx_info = gui_app.inspect_vsdx_for_images(output.vsdx)
            self_report = load_json(output.self_check_json)
            row.update(
                {
                    "ok": True,
                    "run_dir": str(output.run_dir),
                    "attempt_dir": str(output.attempt_dir),
                    "source_image": str(output.source_image),
                    "scene": str(output.scene),
                    "vsdx": str(output.vsdx),
                    "png": str(output.png),
                    "svg": str(output.svg),
                    "self_check_json": str(output.self_check_json),
                    "self_check_png": str(output.self_check_png),
                    "quality_json": str(output.quality_json),
                    "quality_md": str(output.quality_md),
                    "review_manifest": str(output.review_manifest),
                    "download_allowed": bool(output.download_allowed),
                    "quality_status": output.quality_status,
                    "quality_label": output.quality_label,
                    "self_check_passed": bool(output.self_check_passed),
                    "self_check_score": float(output.self_check_score),
                    "selected_mode": output.attempt_dir.name.split("_", 2)[-1],
                    "no_image_embedding": not bool(vsdx_info.get("has_embedded_images")),
                    "vsdx_has_embedded_images": bool(vsdx_info.get("has_embedded_images")),
                    "vsdx_media_count": len(vsdx_info.get("media_parts", [])),
                    "vsdx_foreign_data_refs": len(vsdx_info.get("foreign_data_parts", [])),
                    "self_check_status": self_report.get("status"),
                    "self_check_metrics": self_report.get("metrics", {}),
                    **counts,
                }
            )
            append_log(
                log_path,
                (
                    f"DONE {index}/{len(files)} {image_path.name} "
                    f"status={row['quality_status']} score={row['self_check_score']:.3f} "
                    f"download={row['download_allowed']} media={row['vsdx_media_count']} "
                    f"tiles={row['image_tiles']} icons={row['icon_regions']}"
                ),
            )
        except Exception as exc:
            row.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=10)})
            append_log(log_path, f"FAIL {index}/{len(files)} {image_path.name}: {exc!r}")
        finally:
            row["logs"] = item_logs[-120:]
            rows.append(row)
            summary = summarize(rows, batch_dir, source_dir)
            (batch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            make_contact_sheet(rows, batch_dir / "self_check_contact_sheet.png")

    summary = summarize(rows, batch_dir, source_dir)
    (batch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(rows, batch_dir / "self_check_contact_sheet.png")
    append_log(log_path, "SUMMARY " + json.dumps({k: summary[k] for k in summary if k != "rows"}, ensure_ascii=False))
    return 0 if summary["failed_count"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GUI-equivalent no-raster self-check workflow on a folder of images.")
    parser.add_argument("--source-dir", type=Path, default=default_material_dir())
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "work" / "workflow_check" / time.strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_batch(args.source_dir.resolve(), args.batch_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
