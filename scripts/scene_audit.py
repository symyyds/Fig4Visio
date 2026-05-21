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
    node_style,
    normalize_scene_coordinates,
    resolve_profile,
)
from scene_validate import (
    CONTAINER_TYPES,
    base_node_id,
    container_for_point,
    edge_point,
    edge_endpoint_role,
    expanded_box,
    infer_containers,
    is_background_node,
    is_passive_loop_frame,
    font_validation_warnings,
    loss_feedback_stub_issue,
    node_box,
    node_center,
    node_text_for_font,
    path_bounds,
    point_in_box,
    polyline_intersects_box_bbox,
    scene_looks_like_gan_tfr,
    segment_has_diagonal,
    segment_intersects_box_interior,
    terminal_tangent_issue,
    text_has_compact_loss_notation,
    text_has_raw_loss_subscript,
    tfr_panel_layout_issues,
)
from font_utils import font_resolution_for_style

CURVED_EDGE_TYPES = {"curved_arrow", "loop_arrow"}
FEEDBACK_TOKENS = {"loss", "backprop", "feedback", "gradient", "penalty", "adv", "rec"}


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


def looks_like_vector_formula(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(char in text for char in "⎡⎢⎣⎤⎥⎦"):
        return True
    if any(word in text.lower() for word in {"loss", "penalty", "reconstruction", "adversarial", "gradient"}):
        return False
    if len(lines) < 2 or len(lines) > 6:
        return False
    underscored = [line for line in lines if "_" in line]
    if len(underscored) < 2:
        return False
    return all(" " not in line and len(line) <= 24 for line in underscored)


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
    page = scene.get("page", {}) if isinstance(scene.get("page"), dict) else {}
    nodes = scene.get("nodes", [])
    edges = scene.get("edges", [])
    nodes_by_id = {node["id"]: node for node in nodes if node.get("id")}
    node_types_by_id = {node["id"]: node.get("type") for node in nodes if node.get("id")}
    warnings: list[str] = []
    containers_by_node = infer_containers(nodes_by_id, node_types_by_id, warnings)
    containers = [node for node in nodes if node.get("type") in CONTAINER_TYPES]

    route_cache: dict[str, list[tuple[float, float]]] = {}
    edge_container_cache: dict[str, tuple[str | None, str | None, str | None, str | None]] = {}
    rebuild_items: list[str] = []
    audit_items: list[str] = []
    typography_items: list[str] = []
    resolved_font_counts: dict[str, int] = {}
    gan_tfr_context = scene_looks_like_gan_tfr(scene, nodes_by_id)
    metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
    exact_mode = str(metadata.get("fidelity", metadata.get("reconstruction_mode", ""))).lower() in {"exact", "strict", "replica", "reconstruction", "1:1"}

    for edge in edges:
        edge_id = str(edge.get("id", "<missing-id>"))
        edge_container_cache[edge_id] = edge_containers(edge, nodes_by_id, node_types_by_id, containers_by_node)
        try:
            style = edge_style(edge, component_map, profile)
            route_cache[edge_id] = edge_route_points(edge, style, nodes_by_id)
        except Exception as exc:
            audit_items.append(f"- [ ] Route for `{edge_id}` could not be computed: {exc}")

    for node in nodes:
        if node.get("id") and node.get("type") in component_map.get("node_types", {}):
            try:
                _, style, _ = node_style(node, component_map, profile)
                text = node_text_for_font(node)
                resolution = font_resolution_for_style(style, text)
                if resolution.resolved:
                    resolved_font_counts[resolution.resolved] = resolved_font_counts.get(resolution.resolved, 0) + 1
                for issue in font_validation_warnings(node, style, exact_mode=exact_mode):
                    rebuild_prefix = (
                        "[REBUILD] "
                        if "records source font" in issue and "installed as" in issue and "effective render font" in issue
                        else ""
                    )
                    typography_items.append(f"- [ ] {rebuild_prefix}{issue}")
            except Exception as exc:
                typography_items.append(f"- [ ] Font style for `{node.get('id')}` could not be resolved: {exc}")
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
        if node_type == "process_box":
            style = node.get("style", {}) if isinstance(node.get("style"), dict) else {}
            if style.get("line_dash") in {"dash", "dot", "long_dash"} and not str(node.get("text", "")).strip():
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{node.get('id')}` is a dashed empty `process_box`; use `dashed_region` for annotation frames and keep labels as separate text."
                )
        if node_type == "ellipse_node":
            node_id = str(node.get("id", "")).lower()
            if not str(node.get("text", "")).strip() and any(token in node_id for token in {"outer", "loop", "cycle"}):
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{node.get('id')}` looks like a passive ellipse used as a training/cycle loop. Use `loop_arrow`/`curved_arrow` paths for visible flow direction instead of an ellipse plus detached arrowheads."
                )
        if node_type == "text_block":
            text = str(node.get("text", ""))
            if looks_like_vector_formula(text):
                audit_items.append(
                    f"- [ ] `{node.get('id')}` looks like a vector/matrix formula but uses `text_block`; use `math_vector` for aligned brackets and entries."
                )
            if text_has_raw_loss_subscript(text):
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{node.get('id')}` uses raw underscore loss notation; use `math_text` or explicit text runs so `L_adv`/`L_rec` render as subscript-style formulas."
                )
        if node_type in {"math_text", "loss_region"}:
            text_parts: list[str] = []
            if node.get("text"):
                text_parts.append(str(node.get("text")))
            formulas = node.get("formulas", node.get("lines"))
            if isinstance(formulas, list):
                text_parts.extend(str(item) for item in formulas)
            elif isinstance(formulas, str):
                text_parts.append(formulas)
            if any(text_has_compact_loss_notation(part) for part in text_parts):
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{node.get('id')}` uses compact loss notation like `Ladv`/`Lrec`; normalize to `L_adv`/`L_rec` so `math_text` can render subscripts."
                )
        if node_type == "image_tile":
            node_id = str(node.get("id", "")).lower()
            asset_ref = str(node.get("asset_ref", "")).lower()
            if any(token in node_id or token in asset_ref for token in {"quality_head", "extractor", "aggregation_quality"}):
                audit_items.append(
                    f"- [ ] `{node.get('id')}` is a raster tile for a paper wedge/module; prefer editable `trapezoid_node`/`polygon_node` unless this is an intentional fidelity-speed tradeoff."
                )

    for issue in tfr_panel_layout_issues(nodes_by_id, node_types_by_id):
        audit_items.append(f"- [ ] {issue}")

    for edge in edges:
        edge_id = str(edge.get("id", "<missing-id>"))
        source_id, target_id, source_container, target_container = edge_container_cache.get(edge_id, (None, None, None, None))
        source_type = node_types_by_id.get(source_id or "")
        target_type = node_types_by_id.get(target_id or "")
        points = route_cache.get(edge_id, [])
        diagonal = any(segment_has_diagonal(start, end) for start, end in zip(points, points[1:]))
        edge_type = str(edge.get("type", ""))

        if gan_tfr_context:
            from_role = edge_endpoint_role(edge, "from", nodes_by_id, node_types_by_id)
            to_role = edge_endpoint_role(edge, "to", nodes_by_id, node_types_by_id)
            if from_role == "discriminator" and to_role == "generated":
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{edge_id}` appears reversed for a GAN/TFR diagram: Generated/Reconstructed TFR should feed into Discriminator, not Discriminator into Generated."
                )

        if diagonal and not edge.get("allow_diagonal") and edge_type not in {"fork_connector", "residual_connector", "residual_loop", *CURVED_EDGE_TYPES}:
            audit_items.append(f"- [ ] `{edge_id}` is diagonal; most 1-to-1 paper-flow arrows should be horizontal/vertical.")
        if diagonal and edge.get("allow_diagonal") and edge_type in {"arrow_connector", "dynamic_connector"}:
            edge_name = edge_id.lower()
            if any(token in edge_name for token in {"gap", "gmp", "extractor", "quality", "aggregation", "projection", "environment", "spine"}):
                audit_items.append(
                    f"- [ ] `{edge_id}` allows diagonal routing but looks like a paper-flow lane; use `lane_arrow`, forced axis routing, or aligned explicit points."
                )
            if any(token in edge_name for token in FEEDBACK_TOKENS):
                audit_items.append(
                    f"- [ ] `{edge_id}` allows diagonal routing but looks like a loss/backprop feedback path; use `dashed_feedback_path` with explicit orthogonal points."
                )
        if diagonal and edge_type == "lane_arrow":
            audit_items.append(f"- [ ] `{edge_id}` is a `lane_arrow` but still contains a diagonal segment; align endpoints or force the lane axis.")
        if edge_type == "dashed_feedback_path":
            if diagonal:
                rebuild_items.append(f"- [ ] [REBUILD] `{edge_id}` is a `dashed_feedback_path` but still contains a diagonal segment; make the feedback path orthogonal.")
            if edge.get("allow_diagonal"):
                rebuild_items.append(f"- [ ] [REBUILD] `{edge_id}` is a `dashed_feedback_path` but relies on `allow_diagonal`; encode the actual path with explicit points.")
            if not edge.get("allow_region_interior_path"):
                for region in containers:
                    if region.get("type") not in {"dashed_region", "loss_region"}:
                        continue
                    if any(
                        segment_intersects_box_interior(start, end, node_box(region), clearance=0.01)
                        for start, end in zip(points, points[1:])
                    ):
                        rebuild_items.append(
                            f"- [ ] [REBUILD] `{edge_id}` draws through dashed region `{region.get('id')}`; route from a boundary point/port and keep the dashed annotation frame clean."
                        )
                        break
            stub_issue = loss_feedback_stub_issue(edge, points, nodes_by_id, node_types_by_id)
            if stub_issue:
                rebuild_items.append(f"- [ ] [REBUILD] {stub_issue}")
        if edge_type == "loop_arrow" and any(token in edge_id.lower() for token in {"outer", "loop", "cycle"}):
            style = edge_style(edge, component_map, profile)
            curve_mode = str(edge.get("curve_mode", edge.get("curve", style.get("curve_mode", "polyline")))).lower()
            if curve_mode in {"", "polyline", "straight"}:
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{edge_id}` is an outer update loop rendered as `{curve_mode or 'polyline'}`; use `curve_mode: \"smooth\"` and evenly sampled points so it does not look like a polygon border."
                )
            if not (edge.get("semantic_role") or edge.get("loop_role") or edge.get("label_id") or edge.get("loop_label_id")):
                audit_items.append(
                    f"- [ ] `{edge_id}` has no semantic role/label binding; set `semantic_role: \"outer_update_loop\"` and bind the bottom update label so the outer curve reads as process flow, not decoration."
                )
            tangent_issue = terminal_tangent_issue(edge, points)
            if tangent_issue:
                rebuild_items.append(f"- [ ] [REBUILD] {tangent_issue}")
            bounds = path_bounds(points)
            if bounds and isinstance(page.get("width"), (int, float)) and isinstance(page.get("height"), (int, float)):
                x1, y1, x2, y2 = bounds
                page_w = float(page["width"])
                page_h = float(page["height"])
                margin = float(edge.get("page_margin_in", style.get("page_margin_in", 0.0)))
                if x1 < margin or y1 < margin or x2 > page_w - margin or y2 > page_h - margin:
                    rebuild_items.append(
                        f"- [ ] [REBUILD] `{edge_id}` reaches page/export bounds ({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}); keep the loop inside `page_background` so PNG/SVG export does not crop it."
                    )
            label_id = edge.get("label_id") or edge.get("loop_label_id")
            if isinstance(label_id, str) and label_id in nodes_by_id:
                label_box = expanded_box(node_box(nodes_by_id[label_id]), 0.025)
                if polyline_intersects_box_bbox(points, label_box, clearance=0.0):
                    rebuild_items.append(
                        f"- [ ] [REBUILD] `{edge_id}` overlaps its update label `{label_id}`; reshape the bottom arc or move the label so the loop and text do not collide."
                    )
        if edge_type in {"line_segment", "arrow_connector"} and any(token in edge_id.lower() for token in {"outer", "loop", "cycle"}):
            if edge_type == "line_segment" and len(points) >= 4:
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{edge_id}` is part of an outer loop drawn as a plain line segment; combine the loop into one `loop_arrow`/`curved_arrow` so the curve is continuous and the arrowhead follows the tangent."
                )
            elif edge_type == "arrow_connector":
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{edge_id}` looks like a detached arrowhead for an outer loop; put the arrowhead on the `loop_arrow`/`curved_arrow` path instead."
                )

        line_dash = str(edge.get("style", {}).get("line_dash", "")).lower()
        feedback_like = (
            edge_type == "dashed_feedback_path"
            or line_dash in {"dash", "dot", "long_dash"}
            or any(token in edge_id.lower() for token in FEEDBACK_TOKENS)
        )
        if feedback_like and edge_type not in {"dashed_feedback_path", "line_segment"}:
            rebuild_items.append(
                f"- [ ] [REBUILD] `{edge_id}` looks like a dashed/loss/backprop feedback route but uses `{edge_type}`; convert it to `dashed_feedback_path` before further coordinate tuning."
            )
        if feedback_like and edge_type == "line_segment" and str(edge_style(edge, component_map, profile).get("end_arrow", "")).lower() not in {"", "none"}:
            rebuild_items.append(
                f"- [ ] [REBUILD] `{edge_id}` is a dashed feedback-like `line_segment` with an arrowhead; replace fragmented short arrows with one semantic `dashed_feedback_path` or a shared bus."
            )
        if feedback_like and gan_tfr_context:
            target_role = edge_endpoint_role(edge, "to", nodes_by_id, node_types_by_id)
            source_role = edge_endpoint_role(edge, "from", nodes_by_id, node_types_by_id)
            effective_style = edge_style(edge, component_map, profile)
            end_arrow = str(effective_style.get("end_arrow", "")).lower()
            if target_role in {"real_tfr", "generated"} and end_arrow not in {"", "none"}:
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{edge_id}` points a feedback/loss arrow into `{target_role}`; panel/backprop legs should leave TFR panels toward a bus, not terminate at the panel input area."
                )
            if source_role in {"real_tfr", "generated"} and end_arrow not in {"", "none"} and any(token in edge_id.lower() for token in FEEDBACK_TOKENS):
                rebuild_items.append(
                    f"- [ ] [REBUILD] `{edge_id}` starts at a TFR panel but carries an arrowhead; use an arrowless panel-to-bus leg plus separate discriminator stubs."
                )
        if feedback_like and not edge.get("allow_text_overlap"):
            for text_node in nodes:
                text_id = str(text_node.get("id", ""))
                if node_types_by_id.get(text_id) not in {"text_block", "math_text"}:
                    continue
                if not str(text_node.get("text", text_node.get("lines", ""))).strip():
                    continue
                if is_background_node(text_node):
                    continue
                text_box = expanded_box(node_box(text_node), 0.025)
                if any(
                    segment_intersects_box_interior(start, end, text_box, clearance=0.0)
                    for start, end in zip(points, points[1:])
                ):
                    rebuild_items.append(
                        f"- [ ] [REBUILD] `{edge_id}` crosses text node `{text_id}`; reroute dashed/loss/backprop paths around labels instead of nudging text."
                    )
                    break

        if (
            source_container != target_container
            and edge_type not in {"line_segment", "boundary_arrow"}
            and not edge.get("allow_cross_container")
            and not edge.get("allow_direct_cross_container")
        ):
            if source_type != "boundary_port" and target_type != "boundary_port":
                audit_items.append(
                    f"- [ ] `{edge_id}` crosses module boundary from `{source_container}` to `{target_container}` without a `boundary_port`/`boundary_arrow`."
                )

        if edge_type == "line_segment" and source_container != target_container and line_length(points) > 0.35:
            audit_items.append(
                f"- [ ] `{edge_id}` is a long cross-boundary visual line; if the source shows a frame output, replace it with `boundary_arrow`."
            )

    if gan_tfr_context:
        parallel_backprop: list[tuple[str, float, float]] = []
        for edge in edges:
            edge_id = str(edge.get("id", ""))
            if edge.get("type") != "dashed_feedback_path":
                continue
            if not any(token in edge_id.lower() for token in {"backprop", "bottom", "disc", "loss"}):
                continue
            points = route_cache.get(edge_id, [])
            if len(points) < 2:
                continue
            start, end = points[0], points[-1]
            if abs(start[0] - end[0]) <= 0.03 and abs(start[1] - end[1]) > 0.35:
                target_role = edge_endpoint_role(edge, "to", nodes_by_id, node_types_by_id)
                if target_role == "discriminator" or "disc" in edge_id.lower():
                    parallel_backprop.append((edge_id, start[0], start[1]))
        if len(parallel_backprop) >= 3:
            xs = sorted(item[1] for item in parallel_backprop)
            min_spacing = min((b - a for a, b in zip(xs, xs[1:])), default=999.0)
            unbundled = [
                edge_id
                for edge_id, _, _ in parallel_backprop
                if not any(edge.get("id") == edge_id and edge.get("bundle_id") for edge in edges)
            ]
            if min_spacing < 0.18 or unbundled:
                rebuild_items.append(
                    "- [ ] [REBUILD] GAN/TFR backprop arrows contain three or more parallel dashed vertical paths into the discriminator; "
                    "use a shared `merge_bus`/`junction_point` with `bundle_id` and controlled spacing so the bottom loss system reads as one clean feedback bus."
                )

    lines: list[str] = []
    title = scene.get("metadata", {}).get("title", "visiomaster scene")
    lines.append(f"# Visiomaster Audit: {title}")
    lines.append("")
    lines.append(f"- Style profile: `{profile_name}`")
    lines.append(f"- Nodes: {len(nodes)}")
    lines.append(f"- Edges: {len(edges)}")
    visible_count = sum(1 for node in containers if node.get("type") in {"group_container", "dashed_region", "loss_region"})
    dashed_count = sum(1 for node in containers if node.get("type") == "dashed_region")
    loss_count = sum(1 for node in containers if node.get("type") == "loss_region")
    audit_count = sum(1 for node in containers if node.get("type") == "audit_region")
    lines.append(f"- Containers: {len(containers)} (`visible frames`: {visible_count}, `dashed_region`: {dashed_count}, `loss_region`: {loss_count}, `audit_region`: {audit_count})")
    lines.append("")

    lines.append("## Typography Review")
    if resolved_font_counts:
        summary = ", ".join(
            f"`{font}` ({count})"
            for font, count in sorted(resolved_font_counts.items(), key=lambda item: (-item[1], item[0].lower()))
        )
        lines.append(f"- Resolved fonts: {summary}")
    else:
        lines.append("- No text-bearing font usage found.")
    if typography_items:
        lines.extend(typography_items)
    else:
        lines.append("- No obvious font availability or source-font mismatch items found.")
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
    if rebuild_items:
        lines.append("### Rebuild Required")
        lines.extend(rebuild_items)
        lines.append("")
    if audit_items:
        lines.extend(audit_items)
    elif rebuild_items:
        lines.append("- Additional topology review can continue after the rebuild-required items are fixed.")
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
    parser.add_argument(
        "--fail-on-rebuild",
        action="store_true",
        help="Exit non-zero when audit finds [REBUILD] items that require local subsystem reconstruction.",
    )
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
    rebuild_count = report.count("[REBUILD]")
    if args.fail_on_rebuild and rebuild_count:
        print(f"Rebuild-required items: {rebuild_count}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
