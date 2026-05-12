#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scene_to_visio import (
    edge_route_points,
    edge_style,
    load_component_map,
    load_style_profiles,
    normalize_scene_coordinates,
    resolve_profile,
)
from scene_validate import (
    CONTAINER_TYPES,
    base_node_id,
    container_for_point,
    edge_point,
    infer_containers,
    node_box,
    node_center,
    point_in_box,
    segment_has_diagonal,
)


def load_scene(path: Path) -> dict[str, Any]:
    return normalize_scene_coordinates(json.loads(path.read_text(encoding="utf-8")))


def endpoint_node_id(value: Any) -> str | None:
    if isinstance(value, str):
        return base_node_id(value)
    return None


def node_label(node: dict[str, Any]) -> str:
    text = str(node.get("text", node.get("symbol", ""))).replace("\n", "\\n").strip()
    return f"{node.get('id')}[{node.get('type')}]" + (f" `{text}`" if text else "")


def edge_label(edge: dict[str, Any]) -> str:
    return f"{edge.get('id')}[{edge.get('type')}] {edge.get('from', edge.get('from_point'))} -> {edge.get('to', edge.get('to_point'))}"


def line_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    return total


def edge_containers(
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    node_types_by_id: dict[str, str],
    containers_by_node: dict[str, str | None],
) -> tuple[str | None, str | None, str | None, str | None]:
    source_id = endpoint_node_id(edge.get("from"))
    target_id = endpoint_node_id(edge.get("to"))
    source_point = edge_point(edge, "from")
    target_point = edge_point(edge, "to")
    source_container = (
        containers_by_node.get(source_id)
        if source_id
        else container_for_point(source_point, nodes_by_id, node_types_by_id)
    )
    target_container = (
        containers_by_node.get(target_id)
        if target_id
        else container_for_point(target_point, nodes_by_id, node_types_by_id)
    )
    return source_id, target_id, source_container, target_container


def audit_scene(scene: dict[str, Any]) -> str:
    component_map = load_component_map()
    profiles = load_style_profiles()
    profile_name, profile = resolve_profile(scene, profiles, None)
    nodes = scene.get("nodes", [])
    edges = scene.get("edges", [])
    nodes_by_id = {node["id"]: node for node in nodes if node.get("id")}
    node_types_by_id = {node["id"]: node.get("type") for node in nodes if node.get("id")}
    warnings: list[str] = []
    containers_by_node = infer_containers(nodes_by_id, node_types_by_id, warnings)
    containers = [node for node in nodes if node.get("type") in CONTAINER_TYPES]

    route_cache: dict[str, list[tuple[float, float]]] = {}
    edge_container_cache: dict[str, tuple[str | None, str | None, str | None, str | None]] = {}
    audit_items: list[str] = []

    for edge in edges:
        edge_id = str(edge.get("id", "<missing-id>"))
        edge_container_cache[edge_id] = edge_containers(edge, nodes_by_id, node_types_by_id, containers_by_node)
        try:
            style = edge_style(edge, component_map, profile)
            route_cache[edge_id] = edge_route_points(edge, style, nodes_by_id)
        except Exception as exc:
            audit_items.append(f"- [ ] Route for `{edge_id}` could not be computed: {exc}")

    for node in nodes:
        node_type = node.get("type")
        if node_type == "ellipse_node" and str(node.get("text", "")).strip() in {"+", "x", "×", "⊗", "*"}:
            audit_items.append(f"- [ ] `{node.get('id')}` looks like an operator but uses `ellipse_node`; use `operator_node`.")
        if node_type == "feature_map_banded":
            overlays = node.get("overlays", node.get("vertical_bands", []))
            for overlay in overlays or []:
                fill = str(overlay.get("fill", "")).lower() if isinstance(overlay, dict) else ""
                if fill in {"#000000", "#111111", "black"}:
                    audit_items.append(f"- [ ] `{node.get('id')}` uses opaque dark overlay; use `feature_map_grid` if the source is a shaded heatmap.")
                    break

    for edge in edges:
        edge_id = str(edge.get("id", "<missing-id>"))
        source_id, target_id, source_container, target_container = edge_container_cache.get(edge_id, (None, None, None, None))
        source_type = node_types_by_id.get(source_id or "")
        target_type = node_types_by_id.get(target_id or "")
        points = route_cache.get(edge_id, [])
        diagonal = any(segment_has_diagonal(start, end) for start, end in zip(points, points[1:]))
        edge_type = str(edge.get("type", ""))

        if diagonal and not edge.get("allow_diagonal") and edge_type not in {"fork_connector", "residual_connector", "residual_loop"}:
            audit_items.append(f"- [ ] `{edge_id}` is diagonal; most 1-to-1 paper-flow arrows should be horizontal/vertical.")

        if source_container != target_container and edge_type not in {"line_segment", "boundary_arrow"}:
            if source_type != "boundary_port" and target_type != "boundary_port":
                audit_items.append(
                    f"- [ ] `{edge_id}` crosses module boundary from `{source_container}` to `{target_container}` without a `boundary_port`/`boundary_arrow`."
                )

        if edge_type == "line_segment" and source_container != target_container and line_length(points) > 0.35:
            audit_items.append(
                f"- [ ] `{edge_id}` is a long cross-boundary visual line; if the source shows a frame output, replace it with `boundary_arrow`."
            )

    lines: list[str] = []
    title = scene.get("metadata", {}).get("title", "visiomaster scene")
    lines.append(f"# Visiomaster Audit: {title}")
    lines.append("")
    lines.append(f"- Style profile: `{profile_name}`")
    lines.append(f"- Nodes: {len(nodes)}")
    lines.append(f"- Edges: {len(edges)}")
    visible_count = sum(1 for node in containers if node.get("type") == "group_container")
    audit_count = sum(1 for node in containers if node.get("type") == "audit_region")
    lines.append(f"- Containers: {len(containers)} (`group_container`: {visible_count}, `audit_region`: {audit_count})")
    lines.append("")

    if warnings:
        lines.append("## Container Inference Warnings")
        lines.extend(f"- [ ] {warning}" for warning in warnings)
        lines.append("")

    lines.append("## Module Checklist")
    if not containers:
        lines.append("- [ ] No `group_container` or `audit_region` modules found. Complex paper figures should encode visible frames or invisible logical audit regions.")
    for container in containers:
        container_id = str(container.get("id"))
        child_ids = [
            node_id
            for node_id, parent_id in containers_by_node.items()
            if parent_id == container_id and node_id != container_id
        ]
        ingress = []
        egress = []
        internal = []
        for edge in edges:
            edge_id = str(edge.get("id", "<missing-id>"))
            _, _, source_container, target_container = edge_container_cache.get(edge_id, (None, None, None, None))
            if source_container == container_id and target_container == container_id:
                internal.append(edge)
            elif source_container != container_id and target_container == container_id:
                ingress.append(edge)
            elif source_container == container_id and target_container != container_id:
                egress.append(edge)

        lines.append(f"### `{container_id}`")
        if container.get("type") == "audit_region":
            lines.append("- Frame: invisible logical audit region")
        else:
            lines.append(f"- Frame: `{container.get('shape', container.get('container_shape', 'rectangle'))}` `{container.get('line_dash', container.get('style', {}).get('line_dash', 'style/default'))}`")
        lines.append(f"- Children ({len(child_ids)}): " + (", ".join(f"`{child}`" for child in child_ids) if child_ids else "none"))
        lines.append(f"- Incoming edges ({len(ingress)}): " + (", ".join(f"`{edge.get('id')}`" for edge in ingress) if ingress else "none"))
        lines.append(f"- Outgoing edges ({len(egress)}): " + (", ".join(f"`{edge.get('id')}`" for edge in egress) if egress else "none"))
        lines.append(f"- Internal edges ({len(internal)}): " + (", ".join(f"`{edge.get('id')}`" for edge in internal) if internal else "none"))
        lines.append("- [ ] Compare this module against the source crop: frame bounds, child count, labels, colors, and arrow directions.")
        lines.append("- [ ] Check every outgoing edge: does it originate from a component, a boundary, or a bus in the source?")
        lines.append("")

    lines.append("## Topology Review Items")
    if audit_items:
        lines.extend(audit_items)
    else:
        lines.append("- No obvious topology review items found. Still compare the rendered PNG against the source by module.")
    lines.append("")

    lines.append("## Edge Inventory")
    for edge in edges:
        edge_id = str(edge.get("id", "<missing-id>"))
        _, _, source_container, target_container = edge_container_cache.get(edge_id, (None, None, None, None))
        route = edge.get("route", edge.get("style", {}).get("route", "style/default"))
        lines.append(f"- `{edge_id}`: `{edge.get('type')}` `{route}` `{source_container}` -> `{target_container}`")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a module-level audit report for a visiomaster scene.")
    parser.add_argument("scene", help="Path to scene.json")
    parser.add_argument("--output", help="Optional markdown report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    report = audit_scene(load_scene(scene_path))
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = scene_path.with_suffix(".audit.md")
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote audit report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
