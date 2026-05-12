#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_to_visio import (
    edge_route_points,
    edge_style,
    load_style_profiles,
    normalize_scene_coordinates,
    resolve_profile,
)


POINT_TOLERANCE = 0.03
CONTAINER_TOLERANCE = 0.02
ASPECT_RATIO_TOLERANCE = 0.08
CONTAINER_TYPES = {"group_container", "audit_region"}


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


def point_in_box(point: tuple[float, float], box: tuple[float, float, float, float], tolerance: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance


def segment_has_diagonal(start: tuple[float, float], end: tuple[float, float]) -> bool:
    return abs(start[0] - end[0]) > POINT_TOLERANCE and abs(start[1] - end[1]) > POINT_TOLERANCE


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

        no_text_ok_types = {
            "image_tile",
            "grid_matrix",
            "bracket",
            "junction_point",
            "boundary_port",
            "audit_region",
            "merge_bus",
            "boundary_fanout",
            "feature_map_banded",
            "feature_map_grid",
            "wave_signal",
            "classifier_head",
        }
        if strict and "text" not in node and "symbol" not in node and node_type not in no_text_ok_types:
            warnings.append(f"Node `{node_id}` has no `text`.")

    incoming_by_endpoint: dict[str, list[str]] = {}
    outgoing_by_endpoint: dict[str, list[str]] = {}
    profiles = load_style_profiles()
    _, profile = resolve_profile(scene, profiles, None)
    containers_by_node = infer_containers(nodes_by_id, node_types_by_id, warnings)
    validate_alignment(nodes_by_id, node_types_by_id, containers_by_node, warnings)
    validate_boundary_ports(nodes_by_id, node_types_by_id, warnings, errors)

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
            if diagonal_segments and not edge.get("allow_diagonal"):
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
            } and diagonal_segments:
                warnings.append(
                    f"Edge `{edge_id}` is marked `{route_name}` but its computed path is not axis-aligned. "
                    "Align the first/last explicit point with the endpoint or use `hv`/`vh`."
                )

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
                if other_node.get("stack_id") in endpoint_stack_ids:
                    continue
                if node_types_by_id.get(other_id) in {
                    "group_container",
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
