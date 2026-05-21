#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scene_to_visio import load_component_map, load_style_profiles, normalize_scene_coordinates, resolve_profile
from scene_validate import (
    CONTAINER_TYPES,
    estimate_text_box,
    has_valid_box,
    infer_containers,
    node_box,
    node_center,
    safe_node_style,
    validate_scene,
    visible_semantic_nodes,
)


def load_scene(path: Path) -> dict[str, Any]:
    return normalize_scene_coordinates(json.loads(path.read_text(encoding="utf-8")))


def scene_complexity_report(scene: dict[str, Any], strict: bool = False) -> str:
    component_map = load_component_map()
    profiles = load_style_profiles()
    profile_name, profile = resolve_profile(scene, profiles, None)
    errors, warnings = validate_scene(scene, strict=strict)

    nodes = scene.get("nodes", [])
    edges = scene.get("edges", [])
    nodes_by_id = {node["id"]: node for node in nodes if node.get("id")}
    node_types_by_id = {node["id"]: node.get("type") for node in nodes if node.get("id")}
    container_warnings: list[str] = []
    containers_by_node = infer_containers(nodes_by_id, node_types_by_id, container_warnings)
    visible_ids = visible_semantic_nodes(nodes_by_id, node_types_by_id)
    containers = [
        node
        for node in nodes
        if node.get("type") in CONTAINER_TYPES and has_valid_box(node)
    ]

    page = scene.get("page", {}) if isinstance(scene.get("page"), dict) else {}
    page_width = float(page.get("width", 0) or 0)
    page_height = float(page.get("height", 0) or 0)
    aspect_ratio = page_width / page_height if page_height else 0.0

    children_by_container: dict[str, list[str]] = defaultdict(list)
    uncovered: list[str] = []
    for node_id in visible_ids:
        container_id = containers_by_node.get(node_id)
        if container_id:
            children_by_container[str(container_id)].append(node_id)
        else:
            uncovered.append(node_id)

    font_sizes_by_type: dict[str, list[float]] = defaultdict(list)
    text_fit_items: list[str] = []
    for node_id in visible_ids:
        node = nodes_by_id[node_id]
        if not has_valid_box(node):
            continue
        text = str(node.get("text", node.get("symbol", ""))).strip()
        style = safe_node_style(node, component_map, profile)
        font_size = style.get("font_size_pt")
        if isinstance(font_size, (int, float)):
            font_sizes_by_type[str(node_types_by_id.get(node_id))].append(float(font_size))
        if text:
            estimated_w, estimated_h = estimate_text_box(text, float(font_size or 12))
            x1, y1, x2, y2 = node_box(node)
            available_w = max(0.0, (x2 - x1) - 0.10)
            available_h = max(0.0, (y2 - y1) - 0.10)
            if available_w and available_h and (estimated_w > available_w * 1.18 or estimated_h > available_h * 1.15):
                text_fit_items.append(
                    f"`{node_id}` {estimated_w:.2f}x{estimated_h:.2f} in estimated vs {available_w:.2f}x{available_h:.2f} in"
                )

    dense_regions = [
        (container_id, child_ids)
        for container_id, child_ids in children_by_container.items()
        if len(child_ids) > 18
    ]
    cross_region_edges = 0
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        source_id = source.split(":", 1)[0] if isinstance(source, str) else None
        target_id = target.split(":", 1)[0] if isinstance(target, str) else None
        if source_id and target_id and containers_by_node.get(source_id) != containers_by_node.get(target_id):
            cross_region_edges += 1

    lines: list[str] = []
    title = scene.get("metadata", {}).get("title", "visiomaster scene")
    lines.append(f"# Visiomaster Complexity Report: {title}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Style profile: `{profile_name}`")
    lines.append(f"- Page: {page_width:.2f} x {page_height:.2f} in, aspect {aspect_ratio:.2f}")
    lines.append(f"- Visible semantic nodes: {len(visible_ids)}")
    lines.append(f"- Edges: {len(edges)}")
    lines.append(f"- Regions: {len(containers)}")
    lines.append(f"- Region-covered visible nodes: {len(visible_ids) - len(uncovered)}/{len(visible_ids)}")
    lines.append(f"- Cross-region edges: {cross_region_edges}")
    lines.append(f"- Validation warnings: {len(warnings)}")
    lines.append(f"- Validation errors: {len(errors)}")
    lines.append("")

    lines.append("## Recommended Build Mode")
    if len(visible_ids) >= 32 or len(edges) >= 35 or aspect_ratio >= 2.2:
        lines.append("- Use `region_first` or `tiled_subscenes`: rebuild each logical module/crop, validate it, then assemble the full-page scene.")
        lines.append("- Add invisible `audit_region` boxes for source areas that do not have visible dashed frames.")
        lines.append("- Freeze shared style tokens before assembly: body font, small label font, operator font, frame title font, and arrow weight.")
    else:
        lines.append("- Whole-scene authoring is acceptable, but still run module audit before final Visio render.")
    lines.append("")

    lines.append("## Region Load")
    if containers:
        for container in sorted(containers, key=lambda node: (node.get("y", 0), node.get("x", 0))):
            container_id = str(container.get("id"))
            child_ids = children_by_container.get(container_id, [])
            cx, cy = node_center(container)
            label = "dense" if len(child_ids) > 18 else "ok"
            lines.append(f"- `{container_id}`: {len(child_ids)} visible nodes, center=({cx:.2f}, {cy:.2f}) `{label}`")
    else:
        lines.append("- No regions found.")
    if uncovered:
        preview = ", ".join(f"`{node_id}`" for node_id in uncovered[:12])
        suffix = " ..." if len(uncovered) > 12 else ""
        lines.append(f"- Uncovered visible nodes: {preview}{suffix}")
    lines.append("")

    lines.append("## Font Scale")
    for node_type, sizes in sorted(font_sizes_by_type.items()):
        if not sizes:
            continue
        lines.append(f"- `{node_type}`: {min(sizes):.1f}-{max(sizes):.1f} pt across {len(sizes)} nodes")
    if text_fit_items:
        lines.append("")
        lines.append("## Text Fit Risks")
        for item in text_fit_items[:12]:
            lines.append(f"- {item}")
        if len(text_fit_items) > 12:
            lines.append(f"- {len(text_fit_items) - 12} additional text-fit risks suppressed.")
    lines.append("")

    lines.append("## Dense Region Risks")
    if dense_regions:
        for container_id, child_ids in dense_regions:
            lines.append(f"- `{container_id}` has {len(child_ids)} visible nodes; split this region or create a local subscene.")
    else:
        lines.append("- No region exceeds the default density threshold.")
    lines.append("")

    if errors or warnings or container_warnings:
        lines.append("## Validation Snapshot")
        for error in errors[:12]:
            lines.append(f"- ERROR: {error}")
        for warning in [*container_warnings, *warnings][:24]:
            lines.append(f"- WARN: {warning}")
        if len(errors) > 12 or len(warnings) + len(container_warnings) > 24:
            lines.append("- Additional validation items suppressed; run `scene_validate.py` for the full list.")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a large-figure complexity report for a visiomaster scene.")
    parser.add_argument("scene", help="Path to scene.json")
    parser.add_argument("--output", help="Optional markdown report path")
    parser.add_argument("--strict", action="store_true", help="Pass strict mode through to scene validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    report = scene_complexity_report(load_scene(scene_path), strict=args.strict)
    output_path = Path(args.output).resolve() if args.output else scene_path.with_suffix(".complexity.md")
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote complexity report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
