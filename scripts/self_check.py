#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((gray < 242) | ((saturation > 20) & (value < 252))).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def edge_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return (edges > 0).astype(np.uint8)


def f1_score(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    tp = float(np.logical_and(a_bool, b_bool).sum())
    fp = float(np.logical_and(~a_bool, b_bool).sum())
    fn = float(np.logical_and(a_bool, ~b_bool).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    union = float(np.logical_or(a_bool, b_bool).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum()) / union


def color_similarity(source: np.ndarray, replica: np.ndarray) -> float:
    diff = np.mean(np.abs(source.astype(np.float32) - replica.astype(np.float32))) / 255.0
    return max(0.0, min(1.0, 1.0 - float(diff)))


def ink_balance(source_mask: np.ndarray, replica_mask: np.ndarray) -> float:
    source_ratio = float(source_mask.mean())
    replica_ratio = float(replica_mask.mean())
    if source_ratio == 0 and replica_ratio == 0:
        return 1.0
    ratio = replica_ratio / max(1e-6, source_ratio)
    return max(0.0, min(1.0, 1.0 - abs(math.log(max(1e-6, ratio))) / math.log(6.0)))


def grid_density_similarity(source_mask: np.ndarray, replica_mask: np.ndarray, rows: int = 8, cols: int = 16) -> float:
    def densities(mask: np.ndarray) -> np.ndarray:
        values: list[float] = []
        height, width = mask.shape
        for row in range(rows):
            y1 = int(row * height / rows)
            y2 = int((row + 1) * height / rows)
            for col in range(cols):
                x1 = int(col * width / cols)
                x2 = int((col + 1) * width / cols)
                values.append(float(mask[y1:y2, x1:x2].mean()))
        return np.array(values, dtype=np.float32)

    source_values = densities(source_mask)
    replica_values = densities(replica_mask)
    normalizer = float(source_values.mean()) + 1e-6
    relative_l1 = float(np.mean(np.abs(source_values - replica_values)) / normalizer)
    return max(0.0, min(1.0, 1.0 - relative_l1))


def regional_ink_min_ratio(source_mask: np.ndarray, replica_mask: np.ndarray) -> float:
    height, width = source_mask.shape
    regions = [
        (0, int(width * 0.34)),
        (int(width * 0.34), int(width * 0.72)),
        (int(width * 0.72), width),
    ]
    ratios: list[float] = []
    for x1, x2 in regions:
        source_ratio = float(source_mask[:, x1:x2].mean())
        replica_ratio = float(replica_mask[:, x1:x2].mean())
        if source_ratio < 0.025:
            continue
        ratio = replica_ratio / max(1e-6, source_ratio)
        ratios.append(max(0.0, min(1.0, ratio)))
    return min(ratios) if ratios else 1.0


def failed_rules_for_report(
    *,
    score: float,
    threshold: float,
    edge_f1: float,
    min_edge_f1: float,
    fg_iou: float,
    min_foreground_iou: float,
    grid_density: float,
    min_grid_density_similarity: float,
    regional_ink: float,
    min_regional_ink_ratio: float,
    ink: float,
    min_ink_balance: float,
) -> list[dict[str, Any]]:
    checks = [
        ("score_threshold", "score", score, threshold),
        ("edge_f1", "edge_f1", edge_f1, min_edge_f1),
        ("foreground_iou", "foreground_iou", fg_iou, min_foreground_iou),
        (
            "grid_density_similarity",
            "grid_density_similarity",
            grid_density,
            min_grid_density_similarity,
        ),
        ("regional_ink_min_ratio", "regional_ink_min_ratio", regional_ink, min_regional_ink_ratio),
        ("ink_balance", "ink_balance", ink, min_ink_balance),
    ]
    failures: list[dict[str, Any]] = []
    for rule, metric, value, required in checks:
        if value < required:
            failures.append(
                {
                    "rule": rule,
                    "metric": metric,
                    "value": round(float(value), 4),
                    "required": round(float(required), 4),
                }
            )
    return failures


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate_panel(image: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    pil = Image.fromarray(image)
    header = 54
    canvas = Image.new("RGB", (pil.width, pil.height + header), "white")
    canvas.paste(pil, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="#111111", font=load_font(20))
    if subtitle:
        draw.text((12, 32), subtitle, fill="#555555", font=load_font(14))
    return canvas


def write_contact_sheet(
    source: np.ndarray,
    replica: np.ndarray,
    output: Path,
    *,
    score: float,
    passed: bool,
    max_side_width: int = 760,
) -> None:
    diff = np.mean(np.abs(source.astype(np.float32) - replica.astype(np.float32)), axis=2)
    diff_norm = np.clip(diff * 2.2, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    panels = []
    for title, image in (
        ("source", source),
        ("replica", replica),
        ("difference heatmap", heat),
    ):
        pil = Image.fromarray(image)
        pil.thumbnail((max_side_width, max_side_width), Image.Resampling.LANCZOS)
        panels.append(annotate_panel(np.array(pil), title))

    pad = 14
    footer = 42
    width = sum(panel.width for panel in panels) + pad * (len(panels) + 1)
    height = max(panel.height for panel in panels) + pad * 2 + footer
    canvas = Image.new("RGB", (width, height), "white")
    cursor = pad
    for panel in panels:
        canvas.paste(panel, (cursor, pad))
        cursor += panel.width + pad
    draw = ImageDraw.Draw(canvas)
    verdict = "PASS" if passed else "FAIL"
    draw.text((pad, height - footer + 8), f"self-check: {verdict}, score={score:.3f}", fill="#111111", font=load_font(18))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def compare_images(
    source_path: Path,
    replica_path: Path,
    *,
    output_json: Path | None = None,
    output_png: Path | None = None,
    threshold: float = 0.38,
    min_edge_f1: float = 0.08,
    min_foreground_iou: float = 0.08,
    min_grid_density_similarity: float = 0.32,
    min_regional_ink_ratio: float = 0.12,
    min_ink_balance: float = 0.50,
) -> dict[str, Any]:
    source = read_rgb(source_path)
    replica_raw = read_rgb(replica_path)
    replica = resize_to(replica_raw, (source.shape[1], source.shape[0]))

    source_fg = foreground_mask(source)
    replica_fg = foreground_mask(replica)
    source_edges = edge_mask(source)
    replica_edges = edge_mask(replica)

    fg_iou = iou(source_fg, replica_fg)
    fg_f1 = f1_score(source_fg, replica_fg)
    edge_f1 = f1_score(source_edges, replica_edges)
    color = color_similarity(source, replica)
    ink = ink_balance(source_fg, replica_fg)
    grid_density = grid_density_similarity(source_fg, replica_fg)
    regional_ink = regional_ink_min_ratio(source_fg, replica_fg)
    source_ink = float(source_fg.mean())
    replica_ink = float(replica_fg.mean())
    score = 0.40 * edge_f1 + 0.24 * fg_iou + 0.16 * grid_density + 0.12 * ink + 0.08 * color
    passed = bool(
        score >= threshold
        and edge_f1 >= min_edge_f1
        and fg_iou >= min_foreground_iou
        and grid_density >= min_grid_density_similarity
        and regional_ink >= min_regional_ink_ratio
        and ink >= min_ink_balance
    )
    failed_rules = failed_rules_for_report(
        score=score,
        threshold=threshold,
        edge_f1=edge_f1,
        min_edge_f1=min_edge_f1,
        fg_iou=fg_iou,
        min_foreground_iou=min_foreground_iou,
        grid_density=grid_density,
        min_grid_density_similarity=min_grid_density_similarity,
        regional_ink=regional_ink,
        min_regional_ink_ratio=min_regional_ink_ratio,
        ink=ink,
        min_ink_balance=min_ink_balance,
    )

    report: dict[str, Any] = {
        "schema_version": "0.2",
        "source": str(source_path),
        "replica": str(replica_path),
        "status": "pass" if passed else "fail",
        "passed": passed,
        "score": round(float(score), 4),
        "threshold": threshold,
        "failed_rules": failed_rules,
        "metrics": {
            "foreground_iou": round(float(fg_iou), 4),
            "foreground_f1": round(float(fg_f1), 4),
            "edge_f1": round(float(edge_f1), 4),
            "color_similarity": round(float(color), 4),
            "ink_balance": round(float(ink), 4),
            "grid_density_similarity": round(float(grid_density), 4),
            "regional_ink_min_ratio": round(float(regional_ink), 4),
            "source_ink_ratio": round(source_ink, 4),
            "replica_ink_ratio": round(replica_ink, 4),
        },
        "rules": {
            "threshold": threshold,
            "min_edge_f1": min_edge_f1,
            "min_foreground_iou": min_foreground_iou,
            "min_grid_density_similarity": min_grid_density_similarity,
            "min_regional_ink_ratio": min_regional_ink_ratio,
            "min_ink_balance": min_ink_balance,
        },
        "notes": [
            "This is an automatic screenshot-level gate, not proof of perfect semantic replica.",
            "A failed gate means the rendered PNG is visually too far from the source and should trigger another reconstruction round.",
        ],
    }
    if output_png:
        write_contact_sheet(source, replica, output_png, score=score, passed=passed)
        report["comparison_png"] = str(output_png)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare source and rendered Fig4Visio PNG screenshots.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--replica", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--threshold", type=float, default=0.38)
    parser.add_argument("--min-edge-f1", type=float, default=0.08)
    parser.add_argument("--min-foreground-iou", type=float, default=0.08)
    parser.add_argument("--min-grid-density-similarity", type=float, default=0.32)
    parser.add_argument("--min-regional-ink-ratio", type=float, default=0.12)
    parser.add_argument("--min-ink-balance", type=float, default=0.50)
    args = parser.parse_args()
    report = compare_images(
        Path(args.source),
        Path(args.replica),
        output_json=Path(args.output_json),
        output_png=Path(args.output_png),
        threshold=args.threshold,
        min_edge_f1=args.min_edge_f1,
        min_foreground_iou=args.min_foreground_iou,
        min_grid_density_similarity=args.min_grid_density_similarity,
        min_regional_ink_ratio=args.min_regional_ink_ratio,
        min_ink_balance=args.min_ink_balance,
    )
    print(json.dumps({"status": report["status"], "score": report["score"]}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
