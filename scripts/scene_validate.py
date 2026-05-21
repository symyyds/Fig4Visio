#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from font_utils import CJK_FONT_NAMES, font_resolution_for_style, has_cjk_text, installed_font_match, normalize_font_key
from scene_to_visio import (
    edge_route_points,
    edge_style,
    load_style_profiles,
    node_style,
    normalize_scene_coordinates,
    resolve_profile,
)


POINT_TOLERANCE = 0.03
CONTAINER_TOLERANCE = 0.02
ASPECT_RATIO_TOLERANCE = 0.08
CONTAINER_TYPES = {"group_container", "dashed_region", "loss_region", "audit_region"}
CURVED_EDGE_TYPES = {"curved_arrow", "loop_arrow"}
CONTINUOUS_PATH_EDGE_TYPES = {"curved_arrow", "loop_arrow", "dashed_feedback_path"}
GAN_TEXT_TOKENS = {"gan", "generator", "discriminator", "generated", "reconstructed tfr"}
LOSS_FORMULA_PATTERN = re.compile(r"\bL_([A-Za-z][A-Za-z0-9]*)\b")
COMPACT_LOSS_FORMULA_PATTERN = re.compile(r"\bL\s*_?\s*(adv|rec)\b", re.IGNORECASE)


def load_component_map() -> dict:
    path = Path(__file__).resolve().parents[1] / "templates" / "visio_components.json"
    return json.loads(path.read_text(encoding="utf-8"))


def base_node_id(endpoint: str) -> str:
    return endpoint.split(":", 1)[0]


def edge_point(edge: dict, endpoint_name: str) -> tuple[float, float] | None:
    value = edge.get(f"{endpoint_name}_point")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    return float(value[0]), float(value[1])


def endpoint_side(endpoint: str) -> str | None:
    if ":" not in endpoint:
        return None
    return endpoint.split(":", 1)[1].split("@", 1)[0]


def endpoint_position(endpoint: str) -> float | None:
    if "@" not in endpoint:
        return None
    raw_value = endpoint.rsplit("@", 1)[1]
    try:
        return float(raw_value)
    except ValueError:
        return None


def node_box(node: dict) -> tuple[float, float, float, float]:
    x = float(node["x"])
    y = float(node["y"])
    return x, y, x + float(node["w"]), y + float(node["h"])


def node_center(node: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = node_box(node)
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_area(node: dict) -> float:
    return float(node["w"]) * float(node["h"])


def is_background_node(node: dict) -> bool:
    node_id = str(node.get("id", "")).lower()
    if node.get("type") == "page_background" or "background" in node_id:
        return True
    role = str(node.get("role", node.get("semantic_role", ""))).lower()
    if role in {"background", "page_background", "export_background"}:
        return True
    style = node.get("style", {}) if isinstance(node.get("style"), dict) else {}
    text = str(node.get("text", node.get("symbol", ""))).strip()
    return (
        not text
        and bool(node.get("allow_overlap"))
        and str(style.get("line", "")).lower() == "none"
        and str(style.get("fill", "")).lower() in {"#ffffff", "white"}
        and box_area(node) >= 10.0
    )


def is_passive_loop_frame(node: dict) -> bool:
    node_id = str(node.get("id", "")).lower()
    text = str(node.get("text", node.get("symbol", ""))).strip()
    return (
        node.get("type") == "ellipse_node"
        and not text
        and any(token in node_id for token in {"outer", "loop", "cycle"})
    )


def node_semantic_text(node: dict) -> str:
    return " ".join(
        str(node.get(key, ""))
        for key in ("id", "text", "symbol", "title", "subtitle", "input_label", "semantic_role")
    ).lower()


def node_text_for_font(node: dict) -> str:
    parts: list[str] = []
    for key in ("text", "symbol", "title", "subtitle", "input_label"):
        value = node.get(key)
        if value:
            parts.append(str(value))
    formulas = node.get("formulas", node.get("lines"))
    if isinstance(formulas, list):
        parts.extend(str(item) for item in formulas)
    elif formulas:
        parts.append(str(formulas))
    blocks = node.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict):
                value = block.get("text", block.get("label"))
                if value:
                    parts.append(str(value))
    return "\n".join(parts)


def font_validation_warnings(
    node: dict,
    style: dict[str, Any],
    exact_mode: bool = False,
) -> list[str]:
    node_id = node.get("id", "<missing-id>")
    text = node_text_for_font(node)
    if not text.strip() and node.get("type") not in {"operator_node"}:
        return []

    warnings: list[str] = []
    resolution = font_resolution_for_style(style, text)
    requested = resolution.requested
    if requested and not installed_font_match(requested):
        if resolution.resolved and resolution.resolved != requested:
            warnings.append(
                f"Node `{node_id}` requests font `{requested}`, which is not installed; renderer will use `{resolution.resolved}` via `{resolution.role or 'default'}` fallback."
            )
        else:
            warnings.append(
                f"Node `{node_id}` requests font `{requested}`, which is not installed and has no matching fallback."
            )
    elif requested and resolution.used_fallback and resolution.resolved:
        warnings.append(
            f"Node `{node_id}` requested `{requested}` but renderer resolved `{resolution.resolved}`. Check whether this is an intended alias/fallback."
        )

    source_font = style.get("source_font_family") or node.get("source_font_family")
    if source_font:
        source_match = installed_font_match(source_font)
        if source_match and resolution.resolved and normalize_font_key(source_match) != normalize_font_key(resolution.resolved):
            warnings.append(
                f"Node `{node_id}` records source font `{source_font}`, which is installed as `{source_match}`, but effective render font is `{resolution.resolved}`. Set `font_family`/`font_family_candidates` to use the source font."
            )
        elif not source_match:
            warnings.append(
                f"Node `{node_id}` records source font `{source_font}`, but it is not installed; choose a visually close `font_family_candidates` fallback."
            )

    if has_cjk_text(text) and resolution.resolved and resolution.resolved not in CJK_FONT_NAMES:
        warnings.append(
            f"Node `{node_id}` contains CJK text but resolves to `{resolution.resolved}`; use `font_role: cjk_sans`/`cjk_serif` or a CJK-capable font to avoid Visio font substitution."
        )

    if exact_mode and text.strip() and not (style.get("font_family") or style.get("font_family_candidates") or style.get("font_role")):
        warnings.append(
            f"Node `{node_id}` has text in an exact replica but no explicit font family, candidates, or role after style resolution."
        )
    return warnings


def node_has_role(node: dict, role: str) -> bool:
    text = node_semantic_text(node)
    if role == "discriminator":
        return "discriminator" in text or re.search(r"\bdisc\b", text) is not None
    if role == "generated":
        return "generated" in text or "reconstructed tfr" in text
    if role == "generator":
        return "generator" in text and "discriminator" not in text
    if role == "real_tfr":
        return "real" in text and "tfr" in text
    return role in text


def scene_looks_like_gan_tfr(scene: dict, nodes_by_id: dict[str, dict]) -> bool:
    metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
    corpus = " ".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("notes", "")),
            *[node_semantic_text(node) for node in nodes_by_id.values()],
        ]
    ).lower()
    return (
        ("gan" in corpus or "generator" in corpus)
        and "discriminator" in corpus
        and ("generated" in corpus or "reconstructed tfr" in corpus)
    )


def point_touching_node_id(
    point: tuple[float, float] | None,
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
    tolerance: float = 0.04,
) -> str | None:
    if point is None:
        return None
    candidates: list[dict] = []
    for node_id, node in nodes_by_id.items():
        if not has_valid_box(node):
            continue
        if is_background_node(node):
            continue
        if node_types_by_id.get(node_id) in CONTAINER_TYPES | {"audit_region", "text_block", "junction_point"}:
            continue
        if point_in_box(point, node_box(node), tolerance=tolerance):
            candidates.append(node)
    if not candidates:
        return None
    return min(candidates, key=box_area).get("id")


def edge_endpoint_node_id(
    edge: dict,
    endpoint_name: str,
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
) -> str | None:
    endpoint = edge.get(endpoint_name)
    if isinstance(endpoint, str):
        node_id = base_node_id(endpoint)
        if node_id in nodes_by_id:
            return node_id
    return point_touching_node_id(edge_point(edge, endpoint_name), nodes_by_id, node_types_by_id)


def edge_endpoint_role(
    edge: dict,
    endpoint_name: str,
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
) -> str | None:
    node_id = edge_endpoint_node_id(edge, endpoint_name, nodes_by_id, node_types_by_id)
    if not node_id:
        return None
    node = nodes_by_id[node_id]
    for role in ("discriminator", "generated", "generator", "real_tfr"):
        if node_has_role(node, role):
            return role
    return None


def segment_length(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def turn_angle_degrees(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    len1 = math.hypot(*v1)
    len2 = math.hypot(*v2)
    if len1 <= 1e-9 or len2 <= 1e-9:
        return 0.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
    return math.degrees(math.acos(dot))


def terminal_tangent_issue(edge: dict, route_points: list[tuple[float, float]]) -> str | None:
    if len(route_points) < 4:
        return None
    edge_id = str(edge.get("id", "<missing-id>"))
    if not edge.get("end_tangent_point"):
        return (
            f"Outer loop `{edge_id}` has no `end_tangent_point`. Add an explicit near-end tangent point so the "
            "arrowhead approaches the target smoothly instead of inheriting a kink from the last sampled loop point."
        )
    angle = turn_angle_degrees(route_points[-3], route_points[-2], route_points[-1])
    if angle > 55 and not edge.get("allow_terminal_kink"):
        return (
            f"Outer loop `{edge_id}` has a {angle:.1f} degree turn at the arrowhead. Move `end_tangent_point` "
            "onto the visual approach direction, or mark `allow_terminal_kink: true` only when the source really bends there."
        )
    final_len = segment_length(route_points[-2], route_points[-1])
    prev_len = segment_length(route_points[-3], route_points[-2])
    if prev_len > 0 and (final_len / prev_len < 0.18 or final_len / prev_len > 3.2):
        return (
            f"Outer loop `{edge_id}` has an imbalanced final approach segment. Keep the final tangent segment close "
            "to neighboring segment length so the arrowhead does not look detached or abruptly stretched."
        )
    return None


def axis_overlap(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    axis: str,
) -> float:
    if axis == "x":
        return max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    if axis == "y":
        return max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return 0.0


def feedback_source_region_id(
    edge: dict,
    route_points: list[tuple[float, float]],
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
    region_types: set[str] | None = None,
) -> str | None:
    region_types = region_types or {"dashed_region", "loss_region"}
    source = edge.get("from")
    if isinstance(source, str):
        source_id = base_node_id(source)
        if node_types_by_id.get(source_id) in region_types:
            return source_id
    start = route_points[0] if route_points else edge_point(edge, "from")
    if start is None:
        return None
    containing = [
        node
        for node_id, node in nodes_by_id.items()
        if node_types_by_id.get(node_id) in region_types
        and point_in_box(start, node_box(node), tolerance=CONTAINER_TOLERANCE)
    ]
    if not containing:
        return None
    return min(containing, key=box_area).get("id")


def loss_feedback_stub_issue(
    edge: dict,
    route_points: list[tuple[float, float]],
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
) -> str | None:
    if edge.get("type") != "dashed_feedback_path" or len(route_points) < 2:
        return None
    source_region_id = feedback_source_region_id(edge, route_points, nodes_by_id, node_types_by_id, {"loss_region"})
    if not source_region_id:
        return None
    target_id = edge_endpoint_node_id(edge, "to", nodes_by_id, node_types_by_id)
    if not target_id or target_id == source_region_id or target_id not in nodes_by_id:
        return None
    target_type = node_types_by_id.get(target_id)
    if target_type in CONTAINER_TYPES | {"text_block", "junction_point", "boundary_port", "merge_bus"}:
        return None

    region_box = node_box(nodes_by_id[source_region_id])
    target_box = node_box(nodes_by_id[target_id])
    region_w = max(1e-9, region_box[2] - region_box[0])
    region_h = max(1e-9, region_box[3] - region_box[1])
    target_w = max(1e-9, target_box[2] - target_box[0])
    target_h = max(1e-9, target_box[3] - target_box[1])
    horizontal_overlap = axis_overlap(region_box, target_box, "x")
    vertical_overlap = axis_overlap(region_box, target_box, "y")
    source_center = ((region_box[0] + region_box[2]) / 2, (region_box[1] + region_box[3]) / 2)
    target_center = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    target_endpoint = edge.get("to")
    side = endpoint_side(target_endpoint) if isinstance(target_endpoint, str) else None
    edge_id = str(edge.get("id", "<missing-id>"))

    if horizontal_overlap >= min(region_w, target_w) * 0.25 and target_center[1] >= source_center[1]:
        clean_vertical_stub = (
            side == "top"
            and len(route_points) == 2
            and segment_axis(route_points[0], route_points[-1]) == "vertical"
        )
        if not clean_vertical_stub:
            return (
                f"Loss feedback path `{edge_id}` leaves `{source_region_id}` toward overlapping target `{target_id}` "
                "as a side/L-shaped route. Use short vertical boundary-to-top stubs; otherwise the loss frame reads as "
                "an extra dashed box plus arrow."
            )
    if horizontal_overlap >= min(region_w, target_w) * 0.25 and target_center[1] < source_center[1]:
        clean_vertical_stub = (
            side == "bottom"
            and len(route_points) == 2
            and segment_axis(route_points[0], route_points[-1]) == "vertical"
        )
        if not clean_vertical_stub:
            return (
                f"Loss feedback path `{edge_id}` leaves `{source_region_id}` toward overlapping target `{target_id}` "
                "without a clean vertical boundary stub."
            )
    if vertical_overlap >= min(region_h, target_h) * 0.25 and target_center[0] >= source_center[0]:
        clean_horizontal_stub = (
            side == "left"
            and len(route_points) == 2
            and segment_axis(route_points[0], route_points[-1]) == "horizontal"
        )
        if not clean_horizontal_stub:
            return (
                f"Loss feedback path `{edge_id}` leaves `{source_region_id}` toward side target `{target_id}` "
                "without a clean horizontal boundary stub."
            )
    if vertical_overlap >= min(region_h, target_h) * 0.25 and target_center[0] < source_center[0]:
        clean_horizontal_stub = (
            side == "right"
            and len(route_points) == 2
            and segment_axis(route_points[0], route_points[-1]) == "horizontal"
        )
        if not clean_horizontal_stub:
            return (
                f"Loss feedback path `{edge_id}` leaves `{source_region_id}` toward side target `{target_id}` "
                "without a clean horizontal boundary stub."
            )
    return None


def text_has_raw_loss_subscript(text: str) -> bool:
    lowered = text.lower()
    return (
        bool(LOSS_FORMULA_PATTERN.search(text))
        or bool(COMPACT_LOSS_FORMULA_PATTERN.search(text))
    ) and any(
        token in lowered for token in {"loss", "penalty", "adversarial", "reconstruction", "gradient", "l_", "ladv", "lrec"}
    )


def text_has_compact_loss_notation(text: str) -> bool:
    return bool(COMPACT_LOSS_FORMULA_PATTERN.search(str(text))) and "_" not in str(text)


def segment_bbox_intersects_box(
    start: tuple[float, float],
    end: tuple[float, float],
    box: tuple[float, float, float, float],
    clearance: float = 0.0,
) -> bool:
    lo_x, hi_x = sorted((start[0], end[0]))
    lo_y, hi_y = sorted((start[1], end[1]))
    x1, y1, x2, y2 = box
    return max(lo_x, x1 + clearance) <= min(hi_x, x2 - clearance) and max(lo_y, y1 + clearance) <= min(hi_y, y2 - clearance)


def polyline_intersects_box_bbox(
    points: list[tuple[float, float]],
    box: tuple[float, float, float, float],
    clearance: float = 0.0,
) -> bool:
    return any(segment_bbox_intersects_box(start, end, box, clearance=clearance) for start, end in zip(points, points[1:]))


def path_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def expanded_box(box: tuple[float, float, float, float], padding: float) -> tuple[float, float, float, float]:
    return (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding)


def point_in_box(point: tuple[float, float], box: tuple[float, float, float, float], tolerance: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance


def segment_has_diagonal(start: tuple[float, float], end: tuple[float, float]) -> bool:
    return abs(start[0] - end[0]) > POINT_TOLERANCE and abs(start[1] - end[1]) > POINT_TOLERANCE


def segment_axis(start: tuple[float, float], end: tuple[float, float]) -> str:
    if abs(start[1] - end[1]) <= POINT_TOLERANCE:
        return "horizontal"
    if abs(start[0] - end[0]) <= POINT_TOLERANCE:
        return "vertical"
    return "diagonal"


def route_axes(points: list[tuple[float, float]]) -> set[str]:
    return {segment_axis(start, end) for start, end in zip(points, points[1:])}


def segment_intersects_box_interior(
    start: tuple[float, float],
    end: tuple[float, float],
    box: tuple[float, float, float, float],
    clearance: float = 0.015,
) -> bool:
    x1, y1, x2, y2 = box
    sx, sy = start
    ex, ey = end

    if abs(sy - ey) <= POINT_TOLERANCE:
        y = (sy + ey) / 2
        if not (y1 + clearance < y < y2 - clearance):
            return False
        lo, hi = sorted((sx, ex))
        return max(lo, x1 + clearance) < min(hi, x2 - clearance)

    if abs(sx - ex) <= POINT_TOLERANCE:
        x = (sx + ex) / 2
        if not (x1 + clearance < x < x2 - clearance):
            return False
        lo, hi = sorted((sy, ey))
        return max(lo, y1 + clearance) < min(hi, y2 - clearance)

    return False


def infer_containers(nodes_by_id: dict[str, dict], node_types_by_id: dict[str, str], warnings: list[str]) -> dict[str, str | None]:
    containers = [
        node
        for node in nodes_by_id.values()
        if node_types_by_id.get(node.get("id")) in CONTAINER_TYPES
    ]
    container_ids = {node["id"] for node in containers}
    result: dict[str, str | None] = {}

    for node_id, node in nodes_by_id.items():
        if is_background_node(node):
            result[node_id] = None
            continue
        if node_types_by_id.get(node_id) in CONTAINER_TYPES:
            result[node_id] = None
            continue

        explicit_container = node.get("container_id")
        if explicit_container:
            if explicit_container not in container_ids:
                warnings.append(
                    f"Node `{node_id}` has unknown container_id `{explicit_container}`."
                )
                result[node_id] = None
            else:
                result[node_id] = str(explicit_container)
            continue

        center = node_center(node)
        containing = [
            container
            for container in containers
            if point_in_box(center, node_box(container), tolerance=CONTAINER_TOLERANCE)
        ]
        if not containing:
            result[node_id] = None
        else:
            result[node_id] = min(containing, key=box_area)["id"]

    return result


def container_for_point(point: tuple[float, float] | None, nodes_by_id: dict[str, dict], node_types_by_id: dict[str, str]) -> str | None:
    if point is None:
        return None
    containing = [
        node
        for node in nodes_by_id.values()
        if node_types_by_id.get(node.get("id")) in CONTAINER_TYPES
        and point_in_box(point, node_box(node), tolerance=CONTAINER_TOLERANCE)
    ]
    if not containing:
        return None
    return min(containing, key=box_area)["id"]


def normalize_alignment_axes(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def validate_alignment(
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
    containers_by_node: dict[str, str | None],
    warnings: list[str],
) -> None:
    align_groups: dict[tuple[str, str], list[str]] = {}

    for node_id, node in nodes_by_id.items():
        if node_types_by_id.get(node_id) in CONTAINER_TYPES:
            continue

        tolerance = float(node.get("align_tolerance_in", 0.05))
        for axis in normalize_alignment_axes(node.get("align_to_container")):
            container_id = str(node.get("container_id") or containers_by_node.get(node_id) or "")
            if not container_id or container_id not in nodes_by_id:
                warnings.append(
                    f"Node `{node_id}` requests align_to_container `{axis}` but has no valid container."
                )
                continue
            node_cx, node_cy = node_center(node)
            container_cx, container_cy = node_center(nodes_by_id[container_id])
            if axis == "center_y" and abs(node_cy - container_cy) > tolerance:
                warnings.append(
                    f"Node `{node_id}` is not vertically centered in container `{container_id}` "
                    f"(delta={node_cy - container_cy:.3f} in)."
                )
            elif axis == "center_x" and abs(node_cx - container_cx) > tolerance:
                warnings.append(
                    f"Node `{node_id}` is not horizontally centered in container `{container_id}` "
                    f"(delta={node_cx - container_cx:.3f} in)."
                )
            elif axis not in {"center_x", "center_y"}:
                warnings.append(f"Node `{node_id}` has unsupported align_to_container axis `{axis}`.")

        group_id = node.get("align_group")
        if group_id:
            axis = str(node.get("align_axis", "center_y"))
            align_groups.setdefault((str(group_id), axis), []).append(node_id)

    for (group_id, axis), node_ids in align_groups.items():
        if len(node_ids) < 2:
            continue
        tolerance = max(float(nodes_by_id[node_id].get("align_tolerance_in", 0.05)) for node_id in node_ids)
        centers = [node_center(nodes_by_id[node_id]) for node_id in node_ids]
        values = [center[1] if axis == "center_y" else center[0] for center in centers]
        if axis not in {"center_x", "center_y"}:
            warnings.append(f"Alignment group `{group_id}` has unsupported axis `{axis}`.")
            continue
        if max(values) - min(values) > tolerance:
            warnings.append(
                f"Alignment group `{group_id}` is not aligned on `{axis}` "
                f"(spread={max(values) - min(values):.3f} in; nodes={', '.join(node_ids)})."
            )


def distance_to_container_side(
    point: tuple[float, float],
    container_box: tuple[float, float, float, float],
    side: str,
) -> float:
    x, y = point
    x1, y1, x2, y2 = container_box
    if side == "left":
        return abs(x - x1)
    if side == "right":
        return abs(x - x2)
    if side == "top":
        return abs(y - y1)
    if side == "bottom":
        return abs(y - y2)
    return min(abs(x - x1), abs(x - x2), abs(y - y1), abs(y - y2))


def validate_boundary_ports(
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
    warnings: list[str],
    errors: list[str],
) -> None:
    allowed_sides = {"left", "right", "top", "bottom"}
    allowed_shapes = {"circle", "oval", "dot", "square", "rectangle", "rect", "tick", "line", "none"}
    for node_id, node in nodes_by_id.items():
        if node_types_by_id.get(node_id) != "boundary_port":
            continue

        side = str(node.get("side", "")).lower()
        if side and side not in allowed_sides:
            errors.append(f"Boundary port `{node_id}` has unsupported side `{side}`.")

        shape = str(node.get("shape", "circle")).lower()
        if shape not in allowed_shapes:
            errors.append(f"Boundary port `{node_id}` has unsupported shape `{shape}`.")

        container_id = node.get("container_id")
        if not container_id:
            warnings.append(
                f"Boundary port `{node_id}` has no container_id; use explicit container_id so cross-frame routes stay traceable."
            )
            continue
        if container_id not in nodes_by_id or node_types_by_id.get(container_id) not in CONTAINER_TYPES:
            warnings.append(f"Boundary port `{node_id}` references non-container `{container_id}`.")
            continue

        try:
            center = node_center(node)
            container_box = node_box(nodes_by_id[str(container_id)])
        except Exception:
            continue
        tolerance = float(node.get("boundary_tolerance_in", 0.12))
        if side:
            distance = distance_to_container_side(center, container_box, side)
        else:
            distance = distance_to_container_side(center, container_box, "")
        if distance > tolerance:
            side_text = f" `{side}`" if side else ""
            warnings.append(
                f"Boundary port `{node_id}` is not close to container{side_text} boundary "
                f"(distance={distance:.3f} in)."
            )


def source_aspect_ratio(metadata: dict, warnings: list[str]) -> float | None:
    value = metadata.get("source_aspect_ratio")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)

    source_image = metadata.get("source_image")
    if not source_image:
        return None
    path = Path(str(source_image))
    if not path.exists():
        warnings.append(f"metadata.source_image does not exist: {path}")
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        if height <= 0:
            return None
        return width / height
    except Exception as exc:
        warnings.append(f"Could not read metadata.source_image aspect ratio: {exc}")
        return None


def validate_fidelity_metadata(scene: dict, warnings: list[str]) -> None:
    metadata = scene.get("metadata", {})
    if not isinstance(metadata, dict):
        return
    fidelity = str(metadata.get("fidelity", metadata.get("reconstruction_mode", ""))).lower()
    exact_mode = fidelity in {"exact", "strict", "replica", "reconstruction", "1:1"}
    if not exact_mode:
        return

    if not metadata.get("source_image") and not metadata.get("source_aspect_ratio"):
        warnings.append(
            "Exact reconstruction mode needs metadata.source_image or metadata.source_aspect_ratio; "
            "otherwise page proportions cannot be checked against the source."
        )

    page = scene.get("page", {})
    if not isinstance(page, dict):
        return
    page_width = page.get("width")
    page_height = page.get("height")
    if not isinstance(page_width, (int, float)) or not isinstance(page_height, (int, float)) or page_height <= 0:
        return

    src_ratio = source_aspect_ratio(metadata, warnings)
    if src_ratio is None:
        return
    page_ratio = float(page_width) / float(page_height)
    delta = abs(page_ratio - src_ratio) / src_ratio
    if delta > ASPECT_RATIO_TOLERANCE:
        warnings.append(
            f"Page aspect ratio {page_ratio:.3f} differs from source {src_ratio:.3f} by {delta:.1%}; "
            "exact reconstruction should preserve the source canvas ratio before tuning coordinates."
        )


NON_RENDERED_OR_TINY_TYPES = {
    "page_background",
    "audit_region",
    "junction_point",
    "boundary_port",
    "bracket",
    "merge_bus",
    "boundary_fanout",
}

TEXT_OVERFLOW_SKIP_TYPES = {
    "page_background",
    "audit_region",
    "group_container",
    "loss_region",
    "image_tile",
    "feature_map_banded",
    "feature_map_grid",
    "grid_matrix",
    "bracket",
    "merge_bus",
    "boundary_fanout",
    "classifier_head",
    "wave_signal",
    "modality_spine",
    "math_vector",
    "math_text",
    "tfr_panel",
}


def visible_semantic_nodes(nodes_by_id: dict[str, dict], node_types_by_id: dict[str, str]) -> list[str]:
    return [
        node_id
        for node_id, node_type in node_types_by_id.items()
        if node_type not in CONTAINER_TYPES
        and node_type not in NON_RENDERED_OR_TINY_TYPES
        and not is_background_node(nodes_by_id[node_id])
    ]


def has_valid_box(node: dict[str, Any]) -> bool:
    return (
        all(isinstance(node.get(key), (int, float)) for key in ("x", "y", "w", "h"))
        and float(node.get("w", 0)) > 0
        and float(node.get("h", 0)) > 0
    )


def intersection_area(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def text_width_factor(char: str) -> float:
    if char.isspace():
        return 0.32
    if ord(char) > 255:
        return 0.92
    if char in "ilI.,'`|":
        return 0.28
    if char in "MW@#%&":
        return 0.78
    return 0.54


def estimate_text_box(text: str, font_size_pt: float) -> tuple[float, float]:
    lines = str(text).splitlines() or [str(text)]
    line_widths = [
        sum(text_width_factor(char) for char in line) * font_size_pt / 72.0
        for line in lines
    ]
    width = max(line_widths) if line_widths else 0.0
    height = max(1, len(lines)) * font_size_pt / 72.0 * 1.18
    return width, height


def tfr_panel_layout_issues(nodes_by_id: dict[str, dict], node_types_by_id: dict[str, str]) -> list[str]:
    issues: list[str] = []
    panels = [
        node
        for node_id, node in nodes_by_id.items()
        if node_types_by_id.get(node_id) in {"rounded_process", "process_box"}
        and has_valid_box(node)
        and any(token in node_semantic_text(node) for token in {"real", "generated", "tfr"})
    ]
    grids = [
        node
        for node_id, node in nodes_by_id.items()
        if node_types_by_id.get(node_id) == "grid_matrix"
        and has_valid_box(node)
        and ("tfr" in node_semantic_text(node) or any(point_in_box(node_center(node), node_box(panel), 0.02) for panel in panels))
    ]
    input_labels = [
        node
        for node_id, node in nodes_by_id.items()
        if node_types_by_id.get(node_id) == "text_block"
        and has_valid_box(node)
        and str(node.get("text", "")).strip().lower() == "input"
    ]

    for grid in grids:
        gx1, gy1, gx2, gy2 = node_box(grid)
        gcx, _ = node_center(grid)
        below_labels = [
            label
            for label in input_labels
            if node_box(label)[1] >= gy2
            and abs(node_center(label)[0] - gcx) <= max(float(grid["w"]), float(label["w"])) * 0.75
        ]
        if below_labels:
            label = min(below_labels, key=lambda item: node_box(item)[1] - gy2)
            gap = node_box(label)[1] - gy2
            if gap < 0.08:
                issues.append(
                    f"TFR grid `{grid.get('id')}` is only {gap:.3f} in above `Input`; "
                    "reserve a clear label gap or use a `tfr_panel`/container-local layout before final assembly."
                )

    role_grids: dict[str, dict] = {}
    for grid in grids:
        text = node_semantic_text(grid)
        if "real" in text:
            role_grids["real"] = grid
        if "generated" in text or "reconstructed" in text:
            role_grids["generated"] = grid
    if {"real", "generated"} <= set(role_grids):
        real = role_grids["real"]
        generated = role_grids["generated"]
        rw, rh = float(real["w"]), float(real["h"])
        gw, gh = float(generated["w"]), float(generated["h"])
        if max(abs(rw - gw), abs(rh - gh)) > 0.04 or abs(float(real["y"]) - float(generated["y"])) > 0.04:
            issues.append(
                f"Real/Generated TFR grids are not visually paired (`{real.get('id')}` vs `{generated.get('id')}`); "
                "match grid size, y-position, row/column count, and cell palette before tuning arrows."
            )

    return issues


def safe_node_style(
    node: dict[str, Any],
    component_map: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    try:
        _, style, _ = node_style(node, component_map, profile)
        return style
    except Exception:
        return node.get("style", {}) if isinstance(node.get("style"), dict) else {}


def validate_large_figure_discipline(
    scene: dict,
    nodes_by_id: dict[str, dict],
    node_types_by_id: dict[str, str],
    containers_by_node: dict[str, str | None],
    component_map: dict[str, Any],
    profile: dict[str, Any],
    warnings: list[str],
) -> None:
    nodes = list(nodes_by_id.values())
    edges = scene.get("edges", [])
    page = scene.get("page", {}) if isinstance(scene.get("page"), dict) else {}
    metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
    visible_ids = visible_semantic_nodes(nodes_by_id, node_types_by_id)
    containers = [
        node
        for node in nodes
        if node_types_by_id.get(node.get("id")) in CONTAINER_TYPES
    ]
    page_width = float(page.get("width", 0) or 0)
    page_height = float(page.get("height", 0) or 0)
    aspect_ratio = page_width / page_height if page_height > 0 else 0
    complex_scene = (
        len(visible_ids) >= 32
        or len(edges) >= 35
        or (aspect_ratio >= 2.2 and len(visible_ids) >= 20)
    )

    if complex_scene:
        region_strategy = str(
            metadata.get("region_strategy", metadata.get("large_figure_strategy", ""))
        ).lower()
        if region_strategy not in {"region_first", "tiled_subscenes", "module_first", "section_first"}:
            warnings.append(
                f"Complex scene has {len(visible_ids)} visible nodes and {len(edges)} edges; "
                "set metadata.region_strategy to `region_first`, `tiled_subscenes`, or `module_first`, "
                "then build/review the figure module-by-module before whole-page assembly."
            )

        expected_regions = max(2, min(12, round(len(visible_ids) / 12)))
        if len(containers) < expected_regions:
            warnings.append(
                f"Complex scene has only {len(containers)} group/audit regions; "
                f"add roughly {expected_regions} logical `audit_region`/`group_container` areas so large figures "
                "are reviewed as smaller subscenes instead of one global layout."
            )

        covered_visible_ids = [
            node_id
            for node_id in visible_ids
            if containers_by_node.get(node_id)
        ]
        if visible_ids and len(covered_visible_ids) / len(visible_ids) < 0.75:
            warnings.append(
                f"Only {len(covered_visible_ids)}/{len(visible_ids)} visible nodes are assigned to a region. "
                "For large diagrams, bind nodes with explicit `container_id` or add invisible `audit_region` boxes."
            )

    children_by_container: dict[str, list[str]] = {}
    for node_id, container_id in containers_by_node.items():
        if container_id and node_id in visible_ids:
            children_by_container.setdefault(container_id, []).append(node_id)
    for container_id, child_ids in sorted(children_by_container.items()):
        if len(child_ids) > 18:
            warnings.append(
                f"Region `{container_id}` contains {len(child_ids)} visible nodes; split it into smaller "
                "`audit_region` subregions or create a local subscene first, then assemble it into the full page."
            )

    font_sizes_by_type: dict[str, list[tuple[str, float]]] = {}
    for node_id in visible_ids:
        node = nodes_by_id[node_id]
        node_type = node_types_by_id.get(node_id, "")
        style = safe_node_style(node, component_map, profile)
        font_size = style.get("font_size_pt")
        if isinstance(font_size, (int, float)) and font_size > 0:
            font_sizes_by_type.setdefault(node_type, []).append((node_id, float(font_size)))
    for node_type, values in sorted(font_sizes_by_type.items()):
        if len(values) < 4:
            continue
        sizes = [size for _, size in values]
        if max(sizes) - min(sizes) > 3.0:
            smallest = [node_id for node_id, size in values if size == min(sizes)][:3]
            largest = [node_id for node_id, size in values if size == max(sizes)][:3]
            warnings.append(
                f"Font sizes for `{node_type}` vary from {min(sizes):.1f}pt to {max(sizes):.1f}pt "
                f"(small: {', '.join(smallest)}; large: {', '.join(largest)}). "
                "Large figures should keep each component family on a small role-based font scale."
            )

    overflow_warnings = 0
    for node_id in visible_ids:
        node = nodes_by_id[node_id]
        if not has_valid_box(node):
            continue
        node_type = node_types_by_id.get(node_id, "")
        if node_type in TEXT_OVERFLOW_SKIP_TYPES:
            continue
        text = str(node.get("text", node.get("symbol", ""))).strip()
        if not text:
            continue
        style = safe_node_style(node, component_map, profile)
        font_size = float(style.get("font_size_pt", 12) or 12)
        estimated_w, estimated_h = estimate_text_box(text, font_size)
        padding = float(style.get("text_padding_in", 0.05) or 0.05)
        available_w = max(0.0, float(node.get("w", 0)) - 2 * padding)
        available_h = max(0.0, float(node.get("h", 0)) - 2 * padding)
        if available_w and available_h and (estimated_w > available_w * 1.18 or estimated_h > available_h * 1.15):
            warnings.append(
                f"Text in node `{node_id}` may not fit ({estimated_w:.2f}x{estimated_h:.2f} in estimated "
                f"vs {available_w:.2f}x{available_h:.2f} in available). "
                "Wrap text, enlarge the node, or assign a smaller role font before rendering."
            )
            overflow_warnings += 1
            if overflow_warnings >= 8:
                warnings.append("Additional text-fit warnings suppressed; fix the listed nodes and rerun validation.")
                break

    overlap_warnings = 0
    overlap_ids = [
        node_id
        for node_id in visible_ids
        if has_valid_box(nodes_by_id[node_id])
        if node_types_by_id.get(node_id) not in {"text_block", "wave_signal"}
        and not nodes_by_id[node_id].get("allow_overlap")
    ]
    for index, node_id in enumerate(overlap_ids):
        node = nodes_by_id[node_id]
        box_a = node_box(node)
        area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        for other_id in overlap_ids[index + 1:]:
            other = nodes_by_id[other_id]
            if node.get("stack_id") and node.get("stack_id") == other.get("stack_id"):
                continue
            box_b = node_box(other)
            area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
            overlap = intersection_area(box_a, box_b)
            if overlap <= 0:
                continue
            if min(area_a, area_b) > 0 and overlap / min(area_a, area_b) > 0.20:
                warnings.append(
                    f"Nodes `{node_id}` and `{other_id}` overlap by {overlap:.3f} sq in. "
                    "For intended overlays set `allow_overlap: true`; otherwise fix region-local coordinates."
                )
                overlap_warnings += 1
                if overlap_warnings >= 8:
                    warnings.append("Additional overlap warnings suppressed; fix the listed overlaps and rerun validation.")
                    return


def validate_scene(scene: dict, strict: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        scene = normalize_scene_coordinates(scene)
    except Exception as exc:
        errors.append(f"Coordinate normalization failed: {exc}")
    component_map = load_component_map()
    node_types = set(component_map["node_types"])
    edge_types = set(component_map["edge_types"])
    validate_fidelity_metadata(scene, warnings)
    metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
    fidelity = str(metadata.get("fidelity", metadata.get("reconstruction_mode", ""))).lower()
    exact_mode = fidelity in {"exact", "strict", "replica", "reconstruction", "1:1"}

    required_top = ["version", "page", "nodes", "edges", "assets"]
    for key in required_top:
        if key not in scene:
            errors.append(f"Missing top-level key: {key}")

    page = scene.get("page", {})
    if not isinstance(page, dict):
        errors.append("`page` must be an object.")
    else:
        for dim in ("width", "height"):
            value = page.get(dim)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"`page.{dim}` must be a positive number.")
        if page.get("origin") != "top-left":
            warnings.append("Recommended `page.origin` is `top-left`.")
        if page.get("units") != "in":
            warnings.append("Recommended `page.units` is `in`.")

    nodes = scene.get("nodes", [])
    edges = scene.get("edges", [])
    assets = scene.get("assets", [])

    if not isinstance(nodes, list):
        errors.append("`nodes` must be an array.")
        nodes = []
    if not isinstance(edges, list):
        errors.append("`edges` must be an array.")
        edges = []
    if not isinstance(assets, list):
        errors.append("`assets` must be an array.")
        assets = []

    node_ids: set[str] = set()
    node_types_by_id: dict[str, str] = {}
    nodes_by_id: dict[str, dict] = {}
    asset_ids: set[str] = set()
    edge_ids: set[str] = set()

    for asset in assets:
        asset_id = asset.get("id")
        if not asset_id:
            errors.append("Every asset must have an `id`.")
            continue
        if asset_id in asset_ids:
            errors.append(f"Duplicate asset id: {asset_id}")
        asset_ids.add(asset_id)
        if "path" not in asset:
            warnings.append(f"Asset `{asset_id}` has no `path`.")

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            errors.append("Every node must have an `id`.")
            continue
        if node_id in node_ids:
            errors.append(f"Duplicate node id: {node_id}")
        node_ids.add(node_id)
        nodes_by_id[node_id] = node

        node_type = node.get("type")
        if node_type not in node_types:
            errors.append(f"Unsupported node type `{node_type}` for node `{node_id}`.")
        else:
            node_types_by_id[node_id] = node_type

        dimensions: dict[str, float] = {}
        for key in ("x", "y", "w", "h"):
            value = node.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"Node `{node_id}` is missing numeric `{key}`.")
            elif key in {"w", "h"} and value <= 0:
                errors.append(f"Node `{node_id}` has non-positive `{key}`.")
            elif isinstance(value, (int, float)):
                dimensions[key] = float(value)

        asset_ref = node.get("asset_ref")
        if asset_ref and asset_ref not in asset_ids:
            errors.append(f"Node `{node_id}` references missing asset `{asset_ref}`.")

        has_text = bool(str(node.get("text", node.get("symbol", ""))).strip())
        if (
            node_type in {"process_box", "rounded_process"}
            and not has_text
            and (0 < dimensions.get("w", 1) < 0.06 or 0 < dimensions.get("h", 1) < 0.06)
        ):
            warnings.append(
                f"Node `{node_id}` is an ultra-thin `{node_type}` with no text; "
                "use `bracket` or an edge/connector instead of a fake line box."
            )
        if node_type == "process_box" and not has_text:
            style = node.get("style", {}) if isinstance(node.get("style"), dict) else {}
            if style.get("line_dash") in {"dash", "dot", "long_dash"}:
                warnings.append(
                    f"Node `{node_id}` is an empty dashed `process_box`; use `dashed_region` or `group_container` for visible annotation frames."
                )
        if node_type == "ellipse_node" and not has_text:
            lower_id = str(node_id).lower()
            if any(token in lower_id for token in {"outer", "loop", "cycle"}):
                warnings.append(
                    f"Node `{node_id}` looks like a visible cycle/outer loop frame. "
                    "If it encodes flow direction, rebuild it as `loop_arrow`/`curved_arrow`; "
                    "do not combine a passive ellipse with detached arrowheads."
                )

        if node_type == "bracket":
            orientation = str(node.get("orientation", "right")).lower()
            if orientation not in {"left", "right", "up", "down"}:
                errors.append(
                    f"Bracket `{node_id}` has unsupported orientation `{orientation}`."
                )
            tick_positions = node.get("tick_positions")
            if tick_positions is not None:
                if not isinstance(tick_positions, list):
                    errors.append(f"Bracket `{node_id}` tick_positions must be an array.")
                else:
                    for index, tick in enumerate(tick_positions):
                        if not isinstance(tick, (int, float)) or not 0 <= float(tick) <= 1:
                            errors.append(
                                f"Bracket `{node_id}` tick_positions[{index}] must be a number in [0, 1]."
                            )

        if node_type == "junction_point":
            if dimensions.get("w", 0.0) > 0.2 or dimensions.get("h", 0.0) > 0.2:
                warnings.append(
                    f"Junction point `{node_id}` is larger than usual; keep merge/fan points tiny."
                )

        if node_type == "group_container":
            shape = str(node.get("shape", node.get("container_shape", "rectangle"))).lower()
            if shape not in {"rectangle", "rect", "rounded", "round_rect", "round-rect", "capsule", "pill"}:
                errors.append(f"Group container `{node_id}` has unsupported shape `{shape}`.")
            for key in ("corner_radius_in", "max_rounding_in"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"Group container `{node_id}` {key} must be a non-negative number.")

        if node_type == "dashed_region":
            shape = str(node.get("shape", node.get("container_shape", "rectangle"))).lower()
            if shape not in {"rectangle", "rect", "rounded", "round_rect", "round-rect", "capsule", "pill"}:
                errors.append(f"Dashed region `{node_id}` has unsupported shape `{shape}`.")
            if node.get("text"):
                warnings.append(
                    f"Dashed region `{node_id}` should usually keep labels as separate `text_block` nodes "
                    "so the frame does not compete with internal arrows."
                )
            for key in ("corner_radius_in", "max_rounding_in"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"Dashed region `{node_id}` {key} must be a non-negative number.")

        if node_type == "loss_region":
            shape = str(node.get("shape", node.get("container_shape", "rectangle"))).lower()
            if shape not in {"rectangle", "rect", "rounded", "round_rect", "round-rect", "capsule", "pill"}:
                errors.append(f"Loss region `{node_id}` has unsupported shape `{shape}`.")
            formulas = node.get("formulas", node.get("lines"))
            if formulas is not None and not isinstance(formulas, (list, str)):
                errors.append(f"Loss region `{node_id}` formulas/lines must be an array or string.")
            formula_lines = formulas if isinstance(formulas, list) else str(formulas or "").splitlines()
            if any(text_has_compact_loss_notation(str(line)) for line in formula_lines):
                warnings.append(
                    f"Loss region `{node_id}` uses compact loss notation such as `Ladv`/`Lrec`; normalize formulas to `L_adv`/`L_rec` before rendering."
                )
            title = str(node.get("title", "")).strip()
            if title and has_valid_box(node):
                title_position = str(
                    node.get(
                        "title_position",
                        (node.get("style", {}) if isinstance(node.get("style"), dict) else {}).get("title_position", "header_cutout"),
                    )
                ).lower()
                title_font = float(
                    (node.get("style", {}) if isinstance(node.get("style"), dict) else {}).get(
                        "title_font_size_pt",
                        node.get("title_font_size_pt", 15),
                    )
                )
                title_width, _ = estimate_text_box(title.replace("\n", " "), title_font)
                if title_position not in {"header_cutout", "inside", "top_inside", "inner"}:
                    warnings.append(
                        f"Loss region `{node_id}` title is not protected by a header/inside layout; use `title_position: \"header_cutout\"` "
                        "or `inside` so the dashed frame does not cross the title."
                    )
                if title_width > float(node["w"]) * 1.85:
                    warnings.append(
                        f"Loss region `{node_id}` title is much wider than its frame; enlarge the frame or split the title line before rendering."
                    )
            if node.get("text") and not node.get("title"):
                warnings.append(
                    f"Loss region `{node_id}` should use `title` and `formulas` fields instead of generic text."
                )
            for key in ("title_h_in", "formula_pad_x_in", "formula_pad_y_in"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"Loss region `{node_id}` {key} must be a non-negative number.")

        if node_type == "text_block" and has_text:
            text = str(node.get("text", ""))
            if text_has_raw_loss_subscript(text):
                warnings.append(
                    f"Text block `{node_id}` contains raw underscore loss notation. "
                    "Use `math_text` or explicit text runs so `L_adv`/`L_rec` render with subscript-like formatting."
                )

        if node_type == "math_text":
            text = str(node.get("text", "")).strip()
            lines = node.get("lines")
            if not text and not lines:
                errors.append(f"Math text `{node_id}` needs `text` or `lines`.")
            if lines is not None and not isinstance(lines, list):
                errors.append(f"Math text `{node_id}` lines must be an array.")
            math_lines = lines if isinstance(lines, list) else text.splitlines()
            if any(text_has_compact_loss_notation(str(line)) for line in math_lines):
                warnings.append(
                    f"Math text `{node_id}` uses compact loss notation such as `Ladv`/`Lrec`; normalize to `L_adv`/`L_rec`."
                )
            for key in ("line_gap_in", "subscript_scale", "subscript_offset_in", "segment_gap_in"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"Math text `{node_id}` {key} must be a non-negative number.")

        if node_type == "tfr_panel":
            for key in ("rows", "cols"):
                value = node.get(key, 4 if key == "rows" else 5)
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"TFR panel `{node_id}` {key} must be a positive integer.")
            if not (node.get("title") or node.get("text")):
                warnings.append(f"TFR panel `{node_id}` should set a visible title such as `Real\\nTFR` or `Generated`.")
            cells = node.get("colored_cells", node.get("cells"))
            if cells is not None and not isinstance(cells, list):
                errors.append(f"TFR panel `{node_id}` colored_cells/cells must be an array.")
            for key in ("grid_w", "grid_h", "grid_x", "grid_y", "input_y", "input_gap_in"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"TFR panel `{node_id}` {key} must be a non-negative number.")

        if node_type == "audit_region":
            if node.get("style") and isinstance(node.get("style"), dict):
                line = node["style"].get("line")
                fill = node["style"].get("fill")
                if line not in {None, "none"} or fill not in {None, "none"}:
                    warnings.append(
                        f"Audit region `{node_id}` is intended to be invisible; use `group_container` for visible frames."
                    )

        if node_type == "operator_node" and not has_text:
            warnings.append(
                f"Operator node `{node_id}` has no text; use symbols such as +, x, or tensor product to preserve paper topology."
            )
        if node_type == "operator_node":
            if dimensions.get("w") and dimensions.get("h") and abs(dimensions["w"] - dimensions["h"]) > 0.04:
                warnings.append(
                    f"Operator node `{node_id}` is not square; renderer will center a circle inside the box, but exact replicas should use w ~= h."
                )
            for key in ("symbol_font_size_pt", "symbol_inset_in", "symbol_offset_x_in", "symbol_offset_y_in"):
                value = node.get(key)
                if value is not None and not isinstance(value, (int, float)):
                    errors.append(f"Operator node `{node_id}` {key} must be numeric.")

        if node_type == "boundary_port":
            if dimensions.get("w", 0.0) > 0.22 or dimensions.get("h", 0.0) > 0.22:
                warnings.append(
                    f"Boundary port `{node_id}` is larger than usual; keep ports small and use labels separately."
                )

        if node_type == "wave_signal":
            samples = node.get("samples")
            if samples is not None:
                if not isinstance(samples, list) or not samples:
                    errors.append(f"Wave signal `{node_id}` samples must be a non-empty numeric array.")
                elif not all(isinstance(item, (int, float)) for item in samples):
                    errors.append(f"Wave signal `{node_id}` samples must contain only numbers.")
            cycles = node.get("cycles")
            if cycles is not None and (not isinstance(cycles, (int, float)) or cycles <= 0):
                errors.append(f"Wave signal `{node_id}` cycles must be a positive number.")

        if node_type == "classifier_head":
            orientation = str(node.get("orientation", "horizontal")).lower()
            if orientation not in {"horizontal", "h", "vertical", "v"}:
                errors.append(f"Classifier head `{node_id}` orientation must be horizontal or vertical.")
            blocks = node.get("blocks", node.get("labels", ["AvgPool", "Linear"]))
            if not isinstance(blocks, list) or not blocks:
                errors.append(f"Classifier head `{node_id}` blocks/labels must be a non-empty array.")
            fanout_count = node.get("fanout_count")
            if fanout_count is not None and (not isinstance(fanout_count, int) or fanout_count < 0):
                errors.append(f"Classifier head `{node_id}` fanout_count must be a non-negative integer.")
            for key in ("block_gap_in", "vertical_block_gap_in", "block_width_in", "block_height_in"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"Classifier head `{node_id}` {key} must be a non-negative number.")
            output_mode = str(node.get("output_mode", "")).lower()
            if output_mode and output_mode not in {"none", "internal_fanout", "boundary", "boundary_fanout", "container_boundary", "external"}:
                errors.append(f"Classifier head `{node_id}` has unsupported output_mode `{output_mode}`.")
            if output_mode in {"boundary", "boundary_fanout", "container_boundary", "external"} and fanout_count:
                warnings.append(
                    f"Classifier head `{node_id}` uses boundary output mode; draw output branches with `boundary_fanout`, not internal fanout_count."
                )

        if node_type == "boundary_fanout":
            side = str(node.get("side", "right")).lower()
            if side not in {"left", "right", "top", "bottom"}:
                errors.append(f"Boundary fanout `{node_id}` has unsupported side `{side}`.")
            branch_count = node.get("branch_count")
            positions = node.get("branch_positions", node.get("positions"))
            if branch_count is not None and (not isinstance(branch_count, int) or branch_count <= 0):
                errors.append(f"Boundary fanout `{node_id}` branch_count must be a positive integer.")
            if positions is not None:
                if not isinstance(positions, list) or not positions:
                    errors.append(f"Boundary fanout `{node_id}` branch_positions must be a non-empty numeric array.")
                elif not all(isinstance(item, (int, float)) for item in positions):
                    errors.append(f"Boundary fanout `{node_id}` branch_positions must contain only numbers.")
            if not node.get("container_id"):
                warnings.append(
                    f"Boundary fanout `{node_id}` has no container_id; bind it to the source group_container for faithful frame-edge arrows."
                )

        if node_type in {"stacked_process", "stacked_token"}:
            layers = node.get("layers", node.get("style", {}).get("layers", 4))
            if not isinstance(layers, int) or layers <= 0:
                errors.append(f"Stacked node `{node_id}` must have positive integer `layers`.")

        if node_type == "grid_matrix":
            for key in ("rows", "cols"):
                value = node.get(key)
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"Grid matrix `{node_id}` must have positive integer `{key}`.")

            rows = node.get("rows")
            cols = node.get("cols")
            index_base = int(node.get("index_base", 0))
            cells = node.get("colored_cells", node.get("cells", []))
            if not isinstance(cells, list):
                errors.append(f"Grid matrix `{node_id}` `colored_cells` must be an array.")
            else:
                for index, cell in enumerate(cells):
                    if isinstance(cell, dict):
                        row = cell.get("row")
                        col = cell.get("col")
                    elif isinstance(cell, list) and len(cell) >= 2:
                        row = cell[0]
                        col = cell[1]
                    else:
                        errors.append(f"Grid matrix `{node_id}` cell {index} is invalid.")
                        continue

                    if not isinstance(row, int) or not isinstance(col, int):
                        errors.append(f"Grid matrix `{node_id}` cell {index} row/col must be integers.")
                        continue
                    zero_row = row - index_base
                    zero_col = col - index_base
                    if isinstance(rows, int) and not 0 <= zero_row < rows:
                        errors.append(f"Grid matrix `{node_id}` cell {index} row is out of range.")
                    if isinstance(cols, int) and not 0 <= zero_col < cols:
                        errors.append(f"Grid matrix `{node_id}` cell {index} col is out of range.")

        if node_type == "feature_map_grid":
            rows = node.get("rows")
            cols = node.get("cols", node.get("columns"))
            if rows is not None and (not isinstance(rows, int) or rows <= 0):
                errors.append(f"Feature map grid `{node_id}` rows must be a positive integer.")
            if cols is not None and (not isinstance(cols, int) or cols <= 0):
                errors.append(f"Feature map grid `{node_id}` cols/columns must be a positive integer.")
            for key in ("row_colors", "bands", "column_shades", "row_weights", "row_heights", "column_weights", "column_widths"):
                value = node.get(key)
                if value is not None and not isinstance(value, list):
                    errors.append(f"Feature map grid `{node_id}` {key} must be an array.")
            column_shades = node.get("column_shades")
            if isinstance(column_shades, list):
                for index, value in enumerate(column_shades):
                    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                        errors.append(f"Feature map grid `{node_id}` column_shades[{index}] must be a number in [0, 1].")
            max_shade = node.get("max_shade")
            if max_shade is not None and (not isinstance(max_shade, (int, float)) or not 0 <= float(max_shade) <= 1):
                errors.append(f"Feature map grid `{node_id}` max_shade must be a number in [0, 1].")
            for key in ("show_column_lines", "show_row_lines"):
                value = node.get(key)
                if value is not None and not isinstance(value, bool):
                    errors.append(f"Feature map grid `{node_id}` {key} must be a boolean.")

        if node_type == "polygon_node":
            points = node.get("points")
            if not isinstance(points, list) or len(points) < 3:
                errors.append(f"Polygon node `{node_id}` needs at least three `points`.")
            else:
                for index, point in enumerate(points):
                    if (
                        not isinstance(point, list)
                        or len(point) != 2
                        or not all(isinstance(item, (int, float)) for item in point)
                    ):
                        errors.append(f"Polygon node `{node_id}` points[{index}] must be [x, y] numbers.")

        if node_type == "trapezoid_node":
            orientation = str(node.get("orientation", "right")).lower()
            if orientation not in {"left", "right", "up", "down"}:
                errors.append(f"Trapezoid node `{node_id}` has unsupported orientation `{orientation}`.")
            taper = node.get("taper_ratio", node.get("taper"))
            if taper is not None and (not isinstance(taper, (int, float)) or not 0 <= float(taper) < 0.5):
                errors.append(f"Trapezoid node `{node_id}` taper_ratio must be in [0, 0.5).")

        if node_type == "modality_spine":
            ports = node.get("ports")
            if ports is not None and not isinstance(ports, list):
                errors.append(f"Modality spine `{node_id}` ports must be a list.")
            if isinstance(ports, list):
                for index, port in enumerate(ports):
                    if not isinstance(port, dict):
                        errors.append(f"Modality spine `{node_id}` ports[{index}] must be an object.")
                        continue
                    position = port.get("position", 0.5)
                    if not isinstance(position, (int, float)):
                        errors.append(f"Modality spine `{node_id}` ports[{index}].position must be numeric.")

        if node_type == "math_vector":
            entries = node.get("entries", node.get("rows"))
            text = str(node.get("text", "")).strip()
            if entries is None and not text:
                errors.append(f"Math vector `{node_id}` needs `entries`, `rows`, or text lines.")
            if entries is not None:
                if not isinstance(entries, list) or not entries:
                    errors.append(f"Math vector `{node_id}` entries/rows must be a non-empty array.")
                elif not all(isinstance(item, (str, int, float)) for item in entries):
                    errors.append(f"Math vector `{node_id}` entries must be strings or numbers.")
            for key in ("prefix_w", "gap_in", "bracket_w", "bracket_tick_in", "entry_font_size_pt"):
                value = node.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    errors.append(f"Math vector `{node_id}` {key} must be a non-negative number.")

        no_text_ok_types = {
            "image_tile",
            "grid_matrix",
            "bracket",
            "junction_point",
            "boundary_port",
            "audit_region",
            "dashed_region",
            "loss_region",
            "page_background",
            "merge_bus",
            "boundary_fanout",
            "feature_map_banded",
            "feature_map_grid",
            "wave_signal",
            "classifier_head",
            "polygon_node",
            "trapezoid_node",
            "cuboid_node",
            "modality_spine",
            "math_vector",
            "math_text",
            "tfr_panel",
        }
        if strict and "text" not in node and "symbol" not in node and node_type not in no_text_ok_types:
            warnings.append(f"Node `{node_id}` has no `text`.")

    incoming_by_endpoint: dict[str, list[str]] = {}
    outgoing_by_endpoint: dict[str, list[str]] = {}
    profiles = load_style_profiles()
    _, profile = resolve_profile(scene, profiles, None)
    containers_by_node = infer_containers(nodes_by_id, node_types_by_id, warnings)
    for node in nodes:
        if not node.get("id") or node.get("type") not in node_types:
            continue
        try:
            _, style, _ = node_style(node, component_map, profile)
        except Exception as exc:
            warnings.append(f"Could not resolve style for node `{node.get('id')}` during font validation: {exc}")
            continue
        warnings.extend(font_validation_warnings(node, style, exact_mode))
    validate_alignment(nodes_by_id, node_types_by_id, containers_by_node, warnings)
    validate_boundary_ports(nodes_by_id, node_types_by_id, warnings, errors)
    validate_large_figure_discipline(
        scene,
        nodes_by_id,
        node_types_by_id,
        containers_by_node,
        component_map,
        profile,
        warnings,
    )
    for issue in tfr_panel_layout_issues(nodes_by_id, node_types_by_id):
        warnings.append(issue)

    gan_tfr_context = scene_looks_like_gan_tfr(scene, nodes_by_id)
    for edge in edges:
        edge_id = edge.get("id")
        if not edge_id:
            errors.append("Every edge must have an `id`.")
            continue
        if edge_id in edge_ids:
            errors.append(f"Duplicate edge id: {edge_id}")
        edge_ids.add(edge_id)

        edge_type = edge.get("type")
        if edge_type not in edge_types:
            errors.append(f"Unsupported edge type `{edge_type}` for edge `{edge_id}`.")

        source = edge.get("from")
        target = edge.get("to")
        source_point = edge_point(edge, "from")
        target_point = edge_point(edge, "to")
        if not source and source_point is None:
            errors.append(f"Edge `{edge_id}` must have `from` or `from_point`.")
            continue
        if not target and target_point is None:
            errors.append(f"Edge `{edge_id}` must have `to` or `to_point`.")
            continue

        for endpoint_name, endpoint_value in (("from", source), ("to", target)):
            if endpoint_value is None:
                point_value = edge.get(f"{endpoint_name}_point")
                if edge_point(edge, endpoint_name) is None:
                    errors.append(
                        f"Edge `{edge_id}` {endpoint_name}_point must be [x, y] numbers."
                    )
                elif point_value is None:
                    errors.append(f"Edge `{edge_id}` must have `{endpoint_name}` or `{endpoint_name}_point`.")
                continue
            if not isinstance(endpoint_value, str):
                errors.append(f"Edge `{edge_id}` {endpoint_name} must be a node endpoint string.")
                continue
            node_id = base_node_id(endpoint_value)
            if node_id not in node_ids:
                errors.append(
                    f"Edge `{edge_id}` {endpoint_name} references missing node `{node_id}`."
                )
            elif node_types_by_id.get(node_id) in CONTAINER_TYPES:
                warnings.append(
                    f"Edge `{edge_id}` {endpoint_name} connects to container `{node_id}`; "
                    "containers/audit regions should frame regions, not act as flow endpoints. Use a `junction_point` or explicit border anchor."
                )
            side = endpoint_side(endpoint_value)
            if side and side not in {"left", "right", "top", "bottom", "center"}:
                errors.append(
                    f"Edge `{edge_id}` {endpoint_name} has unsupported side `{side}`."
                )
            if "@" in endpoint_value:
                position = endpoint_position(endpoint_value)
                if position is None or not 0 <= position <= 1:
                    errors.append(
                        f"Edge `{edge_id}` {endpoint_name} endpoint position must use @ratio in [0, 1], for example node:left@0.62."
                    )
                elif side == "center":
                    warnings.append(
                        f"Edge `{edge_id}` {endpoint_name} uses @ratio on center; ratio anchors only affect left/right/top/bottom sides."
                    )

        if isinstance(source, str):
            outgoing_by_endpoint.setdefault(source, []).append(edge_id)
        if isinstance(target, str):
            incoming_by_endpoint.setdefault(target, []).append(edge_id)

        route = edge.get("route") or edge.get("style", {}).get("route")
        if route and route not in {
            "auto",
            "straight",
            "orthogonal",
            "elbow",
            "right_angle",
            "horizontal",
            "vertical",
            "hline",
            "vline",
            "axis_horizontal",
            "axis_vertical",
            "hv",
            "vh",
            "horizontal_then_vertical",
            "vertical_then_horizontal",
        }:
            errors.append(f"Edge `{edge_id}` has unsupported route `{route}`.")

        points = edge.get("points", [])
        if points:
            if not isinstance(points, list):
                errors.append(f"Edge `{edge_id}` `points` must be an array.")
            else:
                for index, point in enumerate(points):
                    if (
                        not isinstance(point, list)
                        or len(point) != 2
                        or not all(isinstance(value, (int, float)) for value in point)
                    ):
                        errors.append(
                            f"Edge `{edge_id}` point {index} must be [x, y] numbers."
                        )
        for tangent_key in ("start_tangent_point", "end_tangent_point"):
            tangent_point = edge.get(tangent_key)
            if tangent_point is not None and (
                not isinstance(tangent_point, list)
                or len(tangent_point) != 2
                or not all(isinstance(value, (int, float)) for value in tangent_point)
            ):
                errors.append(f"Edge `{edge_id}` `{tangent_key}` must be [x, y] numbers.")
        if edge_type in CURVED_EDGE_TYPES and len(points) < 2:
            warnings.append(
                f"Curved edge `{edge_id}` should include several intermediate `points` so it renders as one smooth path. "
                "Do not split a visible loop into separate line and arrowhead edges."
            )

        if (
            edge_type in edge_types
            and (not isinstance(source, str) or base_node_id(source) in nodes_by_id)
            and (not isinstance(target, str) or base_node_id(target) in nodes_by_id)
        ):
            try:
                style = edge_style(edge, component_map, profile)
                route_points = edge_route_points(edge, style, nodes_by_id)
            except Exception as exc:
                warnings.append(f"Edge `{edge_id}` route could not be linted: {exc}")
                continue

            route_name = edge.get("route") or edge.get("style", {}).get("route") or style.get("route") or "auto"
            diagonal_segments = [
                (start, end)
                for start, end in zip(route_points, route_points[1:])
                if segment_has_diagonal(start, end)
            ]
            if diagonal_segments and not edge.get("allow_diagonal") and edge_type not in CURVED_EDGE_TYPES:
                warnings.append(
                    f"Edge `{edge_id}` contains diagonal segment(s); use `hv`/`vh`, aligned explicit points, "
                    "or set `allow_diagonal: true` only for intentional callout lines."
                )
            if route_name in {
                "orthogonal",
                "elbow",
                "right_angle",
                "horizontal",
                "vertical",
                "hline",
                "vline",
                "axis_horizontal",
                "axis_vertical",
                "hv",
                "vh",
                "horizontal_then_vertical",
                "vertical_then_horizontal",
            } and diagonal_segments and edge_type not in CURVED_EDGE_TYPES:
                warnings.append(
                    f"Edge `{edge_id}` is marked `{route_name}` but its computed path is not axis-aligned. "
                    "Align the first/last explicit point with the endpoint or use `hv`/`vh`."
                )

            axes = route_axes(route_points)
            if gan_tfr_context:
                from_role = edge_endpoint_role(edge, "from", nodes_by_id, node_types_by_id)
                to_role = edge_endpoint_role(edge, "to", nodes_by_id, node_types_by_id)
                if from_role == "discriminator" and to_role == "generated":
                    warnings.append(
                        f"GAN/TFR edge `{edge_id}` appears reversed: it goes from Discriminator to Generated. "
                        "For this reconstruction grammar, Generated/Reconstructed TFR should feed into the Discriminator, not the other way around."
                    )

            if edge_type == "lane_arrow":
                if "diagonal" in axes:
                    warnings.append(
                        f"Lane arrow `{edge_id}` contains diagonal segment(s). "
                        "Use `route: \"horizontal\"`/`vertical` or align explicit `from_point`/`to_point`; "
                        "do not use `straight` on slightly mismatched lane y/x values."
                    )
                lane_axis = str(edge.get("lane_axis", edge.get("axis", ""))).lower()
                if lane_axis == "horizontal" and axes - {"horizontal"}:
                    warnings.append(f"Lane arrow `{edge_id}` is declared horizontal but has {sorted(axes)} segment(s).")
                if lane_axis == "vertical" and axes - {"vertical"}:
                    warnings.append(f"Lane arrow `{edge_id}` is declared vertical but has {sorted(axes)} segment(s).")

            if diagonal_segments and edge.get("allow_diagonal") and edge_type in {"arrow_connector", "dynamic_connector"}:
                source_text = str(source or source_point or "")
                target_text = str(target or target_point or "")
                edge_name = str(edge_id).lower()
                likely_lane = any(
                    token in edge_name or token in source_text.lower() or token in target_text.lower()
                    for token in {"gap", "gmp", "extractor", "quality", "aggregation", "projection", "environment", "spine"}
                )
                if likely_lane:
                    warnings.append(
                        f"Edge `{edge_id}` has `allow_diagonal: true` but looks like a paper-flow lane. "
                        "Use `lane_arrow`, forced `horizontal`/`vertical`, side-ratio endpoints, or explicit axis-aligned points instead of accepting a diagonal."
                    )
                likely_feedback = any(
                    token in edge_name or token in source_text.lower() or token in target_text.lower()
                    for token in {"loss", "backprop", "feedback", "gradient", "penalty", "adv", "rec"}
                )
                if likely_feedback:
                    warnings.append(
                        f"Edge `{edge_id}` has `allow_diagonal: true` but looks like a dashed training/feedback path. "
                        "Use `dashed_feedback_path` with explicit orthogonal points, or mark it as an intentional diagonal callout only when the source really is diagonal."
                    )

            if edge_type == "dashed_feedback_path":
                if "diagonal" in axes:
                    warnings.append(
                        f"Dashed feedback path `{edge_id}` contains diagonal segment(s). "
                        "Use explicit orthogonal points for loss/backprop paths."
                    )
                if edge.get("allow_diagonal"):
                    warnings.append(
                        f"Dashed feedback path `{edge_id}` should not rely on `allow_diagonal`; preserve the source path with explicit axis-aligned points."
                    )
                if str(style.get("line_dash", "")).lower() not in {"dash", "dot", "long_dash"}:
                    warnings.append(f"Dashed feedback path `{edge_id}` should use a dashed line style.")
                if not edge.get("allow_region_interior_path"):
                    for region_id, region_node in nodes_by_id.items():
                        if node_types_by_id.get(region_id) not in {"dashed_region", "loss_region"}:
                            continue
                        region_box = node_box(region_node)
                        if any(
                            segment_intersects_box_interior(start, end, region_box, clearance=0.01)
                            for start, end in zip(route_points, route_points[1:])
                        ):
                            warnings.append(
                                f"Dashed feedback path `{edge_id}` draws through dashed region `{region_id}`. "
                                "Keep annotation frames clean: exit through a boundary point/port, then route outside the frame."
                            )
                            break
                stub_issue = loss_feedback_stub_issue(edge, route_points, nodes_by_id, node_types_by_id)
                if stub_issue:
                    warnings.append(stub_issue)

            if edge_type in CURVED_EDGE_TYPES and len(route_points) < 4:
                warnings.append(
                    f"Curved edge `{edge_id}` has too few points for a smooth loop; add sampled curve points or Bezier controls."
                )
            if edge_type == "loop_arrow" and any(token in str(edge_id).lower() for token in {"outer", "loop", "cycle"}):
                curve_mode = str(edge.get("curve_mode", edge.get("curve", style.get("curve_mode", "polyline")))).lower()
                if curve_mode in {"", "polyline", "straight"}:
                    warnings.append(
                        f"Outer loop `{edge_id}` is rendered as `{curve_mode or 'polyline'}`. "
                        "Use `curve_mode: \"smooth\"` and evenly sampled points so the update loop does not look like a decorative polygon border."
                    )
                if not (edge.get("semantic_role") or edge.get("loop_role") or edge.get("label_id")):
                    warnings.append(
                        f"Outer loop `{edge_id}` has no semantic role or label binding. "
                        "Set `semantic_role: \"outer_update_loop\"` and bind the bottom label with `label_id`/`loop_label_id` so it reads as process flow."
                    )
                tangent_issue = terminal_tangent_issue(edge, route_points)
                if tangent_issue:
                    warnings.append(tangent_issue)
                bounds = path_bounds(route_points)
                if bounds and isinstance(page.get("width"), (int, float)) and isinstance(page.get("height"), (int, float)):
                    x1, y1, x2, y2 = bounds
                    page_w = float(page["width"])
                    page_h = float(page["height"])
                    margin = float(edge.get("page_margin_in", style.get("page_margin_in", 0.0)))
                    if x1 < margin or y1 < margin or x2 > page_w - margin or y2 > page_h - margin:
                        warnings.append(
                            f"Outer loop `{edge_id}` reaches page bounds ({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}); "
                            "keep the full loop inside the page/background so export does not crop the curve."
                        )
                label_id = edge.get("label_id") or edge.get("loop_label_id")
                if isinstance(label_id, str) and label_id in nodes_by_id and has_valid_box(nodes_by_id[label_id]):
                    label_box = expanded_box(node_box(nodes_by_id[label_id]), 0.025)
                    if polyline_intersects_box_bbox(route_points, label_box, clearance=0.0):
                        warnings.append(
                            f"Outer loop `{edge_id}` overlaps its label `{label_id}`. Move the label away from the curve "
                            "or reshape the bottom arc before rendering."
                        )
            if edge_type in {"line_segment", "arrow_connector"} and any(token in str(edge_id).lower() for token in {"outer", "loop", "cycle"}):
                if edge_type == "line_segment" and len(route_points) >= 4:
                    warnings.append(
                        f"Edge `{edge_id}` looks like part of a visible loop drawn as a plain line segment. "
                        "Use one `loop_arrow`/`curved_arrow` path so the curve is continuous and arrowheads stay tangent."
                    )
                elif edge_type == "arrow_connector":
                    warnings.append(
                        f"Edge `{edge_id}` looks like a detached loop arrowhead. "
                        "Put the arrowhead on the `loop_arrow`/`curved_arrow` path instead."
                    )

            line_dash = str(style.get("line_dash", "")).lower()
            edge_name = str(edge_id).lower()
            feedback_like = (
                edge_type == "dashed_feedback_path"
                or line_dash in {"dash", "dot", "long_dash"}
                or any(token in edge_name for token in {"loss", "backprop", "feedback", "gradient", "penalty", "adv", "rec"})
            )
            if feedback_like and edge_type not in {"dashed_feedback_path", "line_segment"}:
                warnings.append(
                    f"Edge `{edge_id}` looks like a dashed/loss/backprop feedback route but uses `{edge_type}`. "
                    "Use `dashed_feedback_path` so the path is audited as one continuous feedback route."
                )
            if feedback_like and edge_type == "line_segment" and str(style.get("end_arrow", "")).lower() not in {"", "none"}:
                warnings.append(
                    f"Edge `{edge_id}` is a dashed feedback-like `line_segment` with an arrowhead. "
                    "Use one `dashed_feedback_path` tied to the loss/backprop subsystem; short dashed arrow fragments often become false extra boxes."
                )
            if feedback_like and gan_tfr_context:
                target_role = edge_endpoint_role(edge, "to", nodes_by_id, node_types_by_id)
                source_role = edge_endpoint_role(edge, "from", nodes_by_id, node_types_by_id)
                end_arrow = str(style.get("end_arrow", "")).lower()
                if target_role in {"real_tfr", "generated"} and end_arrow not in {"", "none"}:
                    warnings.append(
                        f"GAN/TFR feedback edge `{edge_id}` points into `{target_role}`. Backprop/loss paths should leave TFR panels "
                        "toward a bus or discriminator, not terminate with an arrowhead at the panel input area."
                    )
                if source_role in {"real_tfr", "generated"} and end_arrow not in {"", "none"} and any(token in edge_name for token in {"backprop", "loss", "rec", "adv"}):
                    warnings.append(
                        f"GAN/TFR feedback edge `{edge_id}` starts at a TFR panel but still has an arrowhead on the far end. "
                        "Panel-to-backprop-bus legs should usually set `end_arrow: none` and let the discriminator stubs carry arrowheads."
                    )
            if feedback_like and not edge.get("allow_text_overlap"):
                for text_node_id, text_node in nodes_by_id.items():
                    if node_types_by_id.get(text_node_id) not in {"text_block", "math_text"}:
                        continue
                    if not str(text_node.get("text", text_node.get("lines", ""))).strip():
                        continue
                    if is_background_node(text_node):
                        continue
                    text_box = expanded_box(node_box(text_node), 0.025)
                    if any(
                        segment_intersects_box_interior(start, end, text_box, clearance=0.0)
                        for start, end in zip(route_points, route_points[1:])
                    ):
                        warnings.append(
                            f"Edge `{edge_id}` crosses text node `{text_node_id}`. "
                            "For exact replicas, reroute dashed/loss/backprop paths around labels instead of nudging text after render."
                        )
                        break

            if edge_type == "boundary_arrow":
                source_node_id = base_node_id(source) if isinstance(source, str) else None
                target_node_id = base_node_id(target) if isinstance(target, str) else None
                source_type = node_types_by_id.get(source_node_id) if source_node_id else None
                target_type = node_types_by_id.get(target_node_id) if target_node_id else None
                if source_type != "boundary_port" and target_type != "boundary_port":
                    warnings.append(
                        f"Boundary arrow `{edge_id}` should start or end at a `boundary_port`; "
                        "use it for frame-edge output, not ordinary component-to-component flow."
                    )
                if route_name not in {"horizontal", "vertical", "hline", "vline", "axis_horizontal", "axis_vertical"}:
                    warnings.append(
                        f"Boundary arrow `{edge_id}` should use a forced axis route such as `horizontal` or `vertical`."
                    )

            source_node_id = base_node_id(source) if isinstance(source, str) else None
            target_node_id = base_node_id(target) if isinstance(target, str) else None
            source_container = (
                containers_by_node.get(source_node_id)
                if source_node_id
                else container_for_point(source_point, nodes_by_id, node_types_by_id)
            )
            target_container = (
                containers_by_node.get(target_node_id)
                if target_node_id
                else container_for_point(target_point, nodes_by_id, node_types_by_id)
            )
            if source_container and target_container and source_container == target_container:
                container_box = node_box(nodes_by_id[source_container])
                if any(
                    not point_in_box(point, container_box, tolerance=CONTAINER_TOLERANCE)
                    for point in route_points
                ):
                    warnings.append(
                        f"Edge `{edge_id}` connects nodes inside `{source_container}` but leaves that container. "
                        "Keep intra-module connectors inside the dashed frame."
                    )
            elif source_container != target_container and edge_type != "line_segment" and not edge.get("allow_cross_container"):
                warnings.append(
                    f"Edge `{edge_id}` crosses container boundary ({source_container} -> {target_container}). "
                    "Split cross-module routes through `junction_point` nodes with `role: boundary_anchor`, "
                    "or mark `allow_cross_container: true` for deliberate callouts."
                )
            if source_container != target_container and edge_type not in {"line_segment", "boundary_arrow"}:
                source_type = node_types_by_id.get(source_node_id) if source_node_id else None
                target_type = node_types_by_id.get(target_node_id) if target_node_id else None
                if (
                    source_type != "boundary_port"
                    and target_type != "boundary_port"
                    and not edge.get("allow_direct_cross_container")
                ):
                    warnings.append(
                        f"Edge `{edge_id}` directly connects components across module boundary "
                        f"({source_container} -> {target_container}). For exact replicas, route through "
                        "`boundary_port`/`boundary_arrow` unless the source visibly connects component-to-component."
                    )

            endpoint_node_ids = {node_id for node_id in (source_node_id, target_node_id) if node_id}
            endpoint_stack_ids = {
                nodes_by_id[node_id].get("stack_id")
                for node_id in endpoint_node_ids
                if nodes_by_id.get(node_id, {}).get("stack_id")
            }
            for other_id, other_node in nodes_by_id.items():
                if other_id in endpoint_node_ids:
                    continue
                if is_background_node(other_node):
                    continue
                if is_passive_loop_frame(other_node):
                    continue
                if other_node.get("stack_id") in endpoint_stack_ids:
                    continue
                if node_types_by_id.get(other_id) in {
                    "group_container",
                    "dashed_region",
                    "loss_region",
                    "audit_region",
                    "junction_point",
                    "boundary_port",
                    "bracket",
                    "text_block",
                    "merge_bus",
                    "boundary_fanout",
                }:
                    continue
                other_box = node_box(other_node)
                if any(
                    segment_intersects_box_interior(start, end, other_box)
                    for start, end in zip(route_points, route_points[1:])
                ):
                    warnings.append(
                        f"Edge `{edge_id}` intersects non-endpoint node `{other_id}`. "
                        "Move it to a bus lane, add a junction/boundary anchor, or add explicit points around the node."
                    )
                    break

    if gan_tfr_context:
        parallel_backprop: list[tuple[str, float, float]] = []
        for edge in edges:
            edge_id = str(edge.get("id", ""))
            if edge.get("type") != "dashed_feedback_path":
                continue
            if not any(token in edge_id.lower() for token in {"backprop", "bottom", "disc", "loss"}):
                continue
            try:
                style = edge_style(edge, component_map, profile)
                route_points = edge_route_points(edge, style, nodes_by_id)
            except Exception:
                continue
            if len(route_points) < 2:
                continue
            start, end = route_points[0], route_points[-1]
            if abs(start[0] - end[0]) <= POINT_TOLERANCE and abs(start[1] - end[1]) > 0.35:
                target_role = edge_endpoint_role(edge, "to", nodes_by_id, node_types_by_id)
                if target_role == "discriminator" or "disc" in edge_id.lower():
                    parallel_backprop.append((edge_id, start[0], start[1]))
        if len(parallel_backprop) >= 3:
            xs = sorted(item[1] for item in parallel_backprop)
            min_spacing = min((b - a for a, b in zip(xs, xs[1:])), default=999)
            unbundled = [edge_id for edge_id, _, _ in parallel_backprop if not any(edge.get("id") == edge_id and edge.get("bundle_id") for edge in edges)]
            if min_spacing < 0.18 or unbundled:
                warnings.append(
                    "GAN/TFR backprop arrows contain three or more parallel dashed vertical paths into the discriminator. "
                    "Use a shared `merge_bus`/`junction_point` with `bundle_id` and controlled spacing so the bottom loss system reads as one clean feedback bus."
                )

    for endpoint, edge_list in incoming_by_endpoint.items():
        node_type = node_types_by_id.get(base_node_id(endpoint))
        if node_type not in {None, "junction_point", "merge_bus"} and len(edge_list) >= 2:
            warnings.append(
                f"Endpoint `{endpoint}` has {len(edge_list)} incoming edges "
                f"({', '.join(edge_list)}); use a tiny `junction_point` when the source figure shows a merged 2-to-1 or many-to-one connector."
            )

    for endpoint, edge_list in outgoing_by_endpoint.items():
        node_type = node_types_by_id.get(base_node_id(endpoint))
        if node_type not in {None, "junction_point", "merge_bus"} and len(edge_list) >= 2:
            warnings.append(
                f"Endpoint `{endpoint}` has {len(edge_list)} outgoing edges "
                f"({', '.join(edge_list)}); use a tiny `junction_point` when the source figure shows a one-to-many fan-out connector."
            )

    return errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a visiomaster scene.json file.")
    parser.add_argument("scene", help="Path to scene.json")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    errors, warnings = validate_scene(scene, strict=args.strict)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Scene is valid: {scene_path}")
    print(
        f"Nodes: {len(scene.get('nodes', []))}, "
        f"Edges: {len(scene.get('edges', []))}, "
        f"Assets: {len(scene.get('assets', []))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
