#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pixel_coordinate_scale(scene: dict[str, Any]) -> tuple[float, float, float, float] | None:
    page = scene.get("page", {})
    metadata = scene.get("metadata", {})
    units = str(page.get("units", metadata.get("coordinate_units", ""))).lower()
    coordinate_space = str(metadata.get("coordinate_space", "")).lower()
    pixel_mode = units in {"px", "pixel", "pixels"} or coordinate_space in {"px", "pixel", "pixels"}
    if not pixel_mode:
        return None

    if units in {"px", "pixel", "pixels"}:
        source_w = page.get("width")
        source_h = page.get("height")
        target_w = page.get("target_width_in", metadata.get("target_width_in", 13.333))
        target_h = page.get("target_height_in", metadata.get("target_height_in"))
    else:
        source_w = page.get("source_width_px", metadata.get("source_width_px"))
        source_h = page.get("source_height_px", metadata.get("source_height_px"))
        target_w = page.get("width")
        target_h = page.get("height")

    if not isinstance(source_w, (int, float)) or not isinstance(source_h, (int, float)):
        raise ValueError("Pixel coordinate scenes require source width/height in pixels.")
    if source_w <= 0 or source_h <= 0:
        raise ValueError("Pixel coordinate source width/height must be positive.")
    if not isinstance(target_w, (int, float)) or target_w <= 0:
        raise ValueError("Pixel coordinate scenes require positive target page width in inches.")
    if target_h is None:
        target_h = float(target_w) * float(source_h) / float(source_w)
    if not isinstance(target_h, (int, float)) or target_h <= 0:
        raise ValueError("Pixel coordinate scenes require positive target page height in inches.")

    return float(target_w) / float(source_w), float(target_h) / float(source_h), float(target_w), float(target_h)


def scale_point(point: list[Any], sx: float, sy: float) -> list[float]:
    return [float(point[0]) * sx, float(point[1]) * sy]


def scale_nested_relative_or_absolute(value: Any, scale: float) -> Any:
    if not isinstance(value, (int, float)):
        return value
    numeric = float(value)
    if -1.0 <= numeric <= 1.0:
        return numeric
    return numeric * scale


def normalize_scene_coordinates(scene: dict[str, Any]) -> dict[str, Any]:
    scale = pixel_coordinate_scale(scene)
    if scale is None:
        return scene

    sx, sy, page_width, page_height = scale
    normalized = copy.deepcopy(scene)
    page = normalized.setdefault("page", {})
    page["width"] = page_width
    page["height"] = page_height
    page["units"] = "in"
    page["origin"] = "top-left"

    for node in normalized.get("nodes", []):
        if all(key in node for key in ("x", "y", "w", "h")):
            node["x"] = float(node["x"]) * sx
            node["y"] = float(node["y"]) * sy
            node["w"] = float(node["w"]) * sx
            node["h"] = float(node["h"]) * sy
        for collection_key in ("notches", "overlays", "vertical_bands"):
            for item in node.get(collection_key, []) or []:
                if not isinstance(item, dict):
                    continue
                for key in ("x", "w"):
                    if key in item:
                        item[key] = scale_nested_relative_or_absolute(item[key], sx)
                for key in ("y", "h"):
                    if key in item:
                        item[key] = scale_nested_relative_or_absolute(item[key], sy)

    for edge in normalized.get("edges", []):
        for key in ("from_point", "to_point"):
            if key in edge:
                edge[key] = scale_point(edge[key], sx, sy)
        if edge.get("points"):
            edge["points"] = [scale_point(point, sx, sy) for point in edge["points"]]

    metadata = normalized.setdefault("metadata", {})
    metadata["normalized_from_units"] = "px"
    metadata["scale_x_in_per_px"] = sx
    metadata["scale_y_in_per_px"] = sy
    return normalized


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_component_map() -> dict[str, Any]:
    return load_json(skill_root() / "templates" / "visio_components.json")


def load_style_profiles() -> dict[str, Any]:
    path = skill_root() / "templates" / "style_profiles.json"
    if not path.exists():
        return {"profiles": {}}
    return load_json(path)


def rgb_formula(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Unsupported color value: {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"RGB({r},{g},{b})"


def hex_rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Unsupported color value: {hex_color}")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def rgb_hex(red: int, green: int, blue: int) -> str:
    return f"#{max(0, min(255, red)):02X}{max(0, min(255, green)):02X}{max(0, min(255, blue)):02X}"


def blend_hex_colors(base: str, overlay: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    br, bg, bb = hex_rgb(base)
    or_, og, ob = hex_rgb(overlay)
    return rgb_hex(
        round(br * (1 - amount) + or_ * amount),
        round(bg * (1 - amount) + og * amount),
        round(bb * (1 - amount) + ob * amount),
    )


def merge_style(*styles: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for style in styles:
        if style:
            merged.update(style)
    return merged


def side_of(endpoint: str, node: dict[str, Any], peer: dict[str, Any]) -> str:
    if ":" in endpoint:
        side = endpoint.split(":", 1)[1]
        return side.split("@", 1)[0]

    node_cx = float(node["x"]) + float(node["w"]) / 2
    node_cy = float(node["y"]) + float(node["h"]) / 2
    peer_cx = float(peer["x"]) + float(peer["w"]) / 2
    peer_cy = float(peer["y"]) + float(peer["h"]) / 2
    dx = peer_cx - node_cx
    dy = peer_cy - node_cy
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "bottom" if dy >= 0 else "top"


def endpoint_position(endpoint: str) -> float | None:
    if "@" not in endpoint:
        return None
    raw_value = endpoint.rsplit("@", 1)[1]
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported endpoint position: {raw_value}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Endpoint position must be in [0, 1]: {raw_value}")
    return value


def resolve_endpoint(endpoint: str, node: dict[str, Any], peer: dict[str, Any]) -> tuple[float, float]:
    page_x = float(node["x"])
    page_y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    side = side_of(endpoint, node, peer)
    position = endpoint_position(endpoint)

    if side == "left":
        return page_x, page_y + height * (0.5 if position is None else position)
    if side == "right":
        return page_x + width, page_y + height * (0.5 if position is None else position)
    if side == "top":
        return page_x + width * (0.5 if position is None else position), page_y
    if side == "bottom":
        return page_x + width * (0.5 if position is None else position), page_y + height
    if side == "center":
        return page_x + width / 2, page_y + height / 2
    raise ValueError(f"Unsupported endpoint side: {side}")


def node_id_from_endpoint(endpoint: str) -> str:
    return endpoint.split(":", 1)[0]


def edge_point(edge: dict[str, Any], endpoint_name: str) -> tuple[float, float] | None:
    value = edge.get(f"{endpoint_name}_point")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Edge `{edge.get('id', '<unknown>')}` {endpoint_name}_point must be [x, y].")
    return float(value[0]), float(value[1])


def fake_node_at(point: tuple[float, float]) -> dict[str, float]:
    return {"x": point[0], "y": point[1], "w": 0.0, "h": 0.0}


def node_center_point(node: dict[str, Any]) -> tuple[float, float]:
    return float(node["x"]) + float(node["w"]) / 2, float(node["y"]) + float(node["h"]) / 2


def resolve_edge_endpoint(
    edge: dict[str, Any],
    endpoint_name: str,
    peer_point: tuple[float, float],
    nodes_by_id: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    point = edge_point(edge, endpoint_name)
    if point is not None:
        return point

    endpoint = edge.get(endpoint_name)
    if not isinstance(endpoint, str):
        raise ValueError(f"Edge `{edge.get('id', '<unknown>')}` requires `{endpoint_name}` or `{endpoint_name}_point`.")
    node = nodes_by_id[node_id_from_endpoint(endpoint)]
    return resolve_endpoint(endpoint, node, fake_node_at(peer_point))


def to_visio_y(page_height: float, scene_y: float) -> float:
    return page_height - scene_y


def try_set_formula(shape: Any, cell_name: str, formula: str) -> None:
    try:
        shape.CellsU(cell_name).FormulaU = formula
    except Exception:
        return


def try_set_result(shape: Any, cell_name: str, value: float | int) -> None:
    try:
        shape.CellsU(cell_name).ResultIU = value
    except Exception:
        return


def try_set_text(shape: Any, text: str) -> None:
    try:
        shape.Text = text
    except Exception:
        return


def arrow_size_value(value: Any, segment_length: float | None = None) -> int:
    if isinstance(value, (int, float)):
        return max(0, min(6, int(value)))
    if isinstance(value, str):
        mapped = {
            "tiny": 0,
            "xsmall": 0,
            "small": 1,
            "medium": 2,
            "normal": 2,
            "large": 3,
            "xlarge": 4,
        }.get(value.lower())
        if mapped is not None:
            return mapped
    if segment_length is not None:
        if segment_length < 0.16:
            return 0
        if segment_length < 0.30:
            return 1
    return 2


def font_style_value(style: dict[str, Any]) -> int:
    value = 0
    weight = style.get("font_weight")
    if isinstance(weight, str) and weight.lower() in {"bold", "semibold", "heavy"}:
        value |= 1
    if isinstance(weight, (int, float)) and weight >= 600:
        value |= 1
    if style.get("font_italic"):
        value |= 2
    return value


def apply_shadow(shape: Any, shadow: dict[str, Any] | bool | None) -> None:
    if not shadow:
        return
    if shadow is True:
        shadow = {}

    try_set_formula(shape, "ShdwPattern", "1")
    try_set_formula(shape, "ShdwForegnd", rgb_formula(str(shadow.get("color", "#000000"))))
    try_set_formula(shape, "ShdwOffsetX", f"{float(shadow.get('offset_x_in', 0.04))} in")
    try_set_formula(shape, "ShdwOffsetY", f"{float(shadow.get('offset_y_in', -0.04))} in")
    transparency = shadow.get("transparency_pct", 78)
    try_set_formula(shape, "ShdwForegndTrans", f"{float(transparency)}%")


def apply_style(shape: Any, style: dict[str, Any]) -> None:
    fill = style.get("fill")
    if fill == "none":
        try_set_result(shape, "FillPattern", 0)
    elif fill:
        try_set_result(shape, "FillPattern", 1)
        try_set_formula(shape, "FillForegnd", rgb_formula(str(fill)))

    fill_transparency = style.get("fill_transparency_pct")
    if fill_transparency is not None:
        try_set_formula(shape, "FillForegndTrans", f"{float(fill_transparency)}%")

    line = style.get("line")
    if line == "none":
        try_set_result(shape, "LinePattern", 0)
    elif line:
        try_set_result(shape, "LinePattern", 1)
        try_set_formula(shape, "LineColor", rgb_formula(str(line)))

    line_transparency = style.get("line_transparency_pct")
    if line_transparency is not None:
        try_set_formula(shape, "LineColorTrans", f"{float(line_transparency)}%")

    line_weight = style.get("line_weight_pt")
    if line_weight is not None:
        try_set_formula(shape, "LineWeight", f"{float(line_weight)} pt")

    line_dash = style.get("line_dash")
    if line_dash == "dash":
        try_set_result(shape, "LinePattern", 2)
    elif line_dash == "dot":
        try_set_result(shape, "LinePattern", 3)
    elif line_dash == "long_dash":
        try_set_result(shape, "LinePattern", 7)

    rounding = style.get("rounding_in")
    if rounding is not None:
        try_set_formula(shape, "Rounding", f"{float(rounding)} in")

    text_color = style.get("text_color")
    if text_color:
        try_set_formula(shape, "Char.Color", rgb_formula(str(text_color)))

    font_size = style.get("font_size_pt")
    if font_size is not None:
        try_set_formula(shape, "Char.Size", f"{float(font_size)} pt")

    font_family = style.get("font_family")
    if font_family:
        try_set_formula(shape, "Char.Font", f'FONT("{font_family}")')

    char_style = font_style_value(style)
    if char_style:
        try_set_result(shape, "Char.Style", char_style)
    elif "font_weight" in style or "font_italic" in style:
        try_set_result(shape, "Char.Style", 0)

    text_angle = style.get("text_angle_deg")
    if text_angle is not None:
        try_set_formula(shape, "TxtAngle", f"{float(text_angle)} deg")

    angle = style.get("angle_deg")
    if angle is not None:
        try_set_formula(shape, "Angle", f"{float(angle)} deg")

    try_set_result(shape, "Para.HorzAlign", int(style.get("text_align", 1)))
    try_set_result(shape, "VerticalAlign", int(style.get("vertical_align", 1)))
    apply_shadow(shape, style.get("shadow"))


def draw_rectangle(page: Any, page_height: float, node: dict[str, Any]) -> Any:
    x1 = float(node["x"])
    y1 = to_visio_y(page_height, float(node["y"]) + float(node["h"]))
    x2 = float(node["x"]) + float(node["w"])
    y2 = to_visio_y(page_height, float(node["y"]))
    return page.DrawRectangle(x1, y1, x2, y2)


def draw_oval(page: Any, page_height: float, node: dict[str, Any]) -> Any:
    x1 = float(node["x"])
    y1 = to_visio_y(page_height, float(node["y"]) + float(node["h"]))
    x2 = float(node["x"]) + float(node["w"])
    y2 = to_visio_y(page_height, float(node["y"]))
    return page.DrawOval(x1, y1, x2, y2)


def draw_text_box(
    page: Any,
    page_height: float,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    style: dict[str, Any],
) -> Any:
    shape = draw_rectangle(page, page_height, {"x": x, "y": y, "w": width, "h": height})
    try_set_text(shape, text)
    apply_style(shape, merge_style(style, {"fill": "none", "line": "none"}))
    return shape


def draw_boundary_port(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    visible_value = node.get("visible", True)
    visible = not (visible_value is False or str(visible_value).lower() in {"false", "0", "no"})
    shape_kind = str(node.get("shape", "circle")).lower()
    if shape_kind == "none":
        visible = False
    port_style = dict(style)
    if not visible:
        port_style = merge_style(port_style, {"fill": "none", "line": "none", "line_weight_pt": 0})

    if shape_kind in {"tick", "line"}:
        x = float(node["x"])
        y = float(node["y"])
        width = float(node["w"])
        height = float(node["h"])
        side = str(node.get("side", "right")).lower()
        if side in {"top", "bottom"}:
            start = (x + width / 2, y)
            end = (x + width / 2, y + height)
        else:
            start = (x, y + height / 2)
            end = (x + width, y + height / 2)
        return draw_line_segment(page, page_height, start, end, merge_style(port_style, {"end_arrow": "none"}))

    if shape_kind in {"square", "rectangle", "rect"}:
        shape = draw_rectangle(page, page_height, node)
    else:
        shape = draw_oval(page, page_height, node)
    apply_style(shape, port_style)
    return shape


def draw_operator_node(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    if node.get("enforce_circle", style.get("enforce_circle", True)):
        size = min(width, height)
        x += (width - size) / 2
        y += (height - size) / 2
        width = size
        height = size
    circle_node = dict(node)
    circle_node.update({"x": x, "y": y, "w": width, "h": height})
    shape = draw_oval(page, page_height, circle_node)
    apply_style(shape, style)

    symbol = str(node.get("symbol", node.get("text", "")))
    if symbol:
        symbol_style = merge_style(
            style,
            {
                "fill": "none",
                "line": "none",
                "font_family": node.get("symbol_font_family", style.get("symbol_font_family", "Cambria Math")),
                "font_size_pt": node.get(
                    "symbol_font_size_pt",
                    style.get("symbol_font_size_pt", max(6, min(width, height) * 72 * 0.58)),
                ),
                "font_weight": node.get("symbol_font_weight", style.get("symbol_font_weight", "regular")),
                "text_align": 1,
                "vertical_align": 1,
            },
        )
        inset = float(node.get("symbol_inset_in", style.get("symbol_inset_in", 0.0)))
        offset_x = float(node.get("symbol_offset_x_in", style.get("symbol_offset_x_in", 0.0)))
        offset_y = float(node.get("symbol_offset_y_in", style.get("symbol_offset_y_in", 0.0)))
        draw_text_box(
            page,
            page_height,
            x + inset + offset_x,
            y + inset + offset_y,
            max(0.01, width - inset * 2),
            max(0.01, height - inset * 2),
            symbol,
            symbol_style,
        )
    return shape


def draw_group_container(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    style = dict(style)
    shape_kind = str(node.get("shape", node.get("container_shape", "rectangle"))).lower()
    width = float(node["w"])
    height = float(node["h"])
    if shape_kind in {"rounded", "round_rect", "round-rect", "capsule", "pill"} and not float(style.get("rounding_in", 0) or 0):
        if node.get("corner_radius_in") is not None:
            style["rounding_in"] = float(node["corner_radius_in"])
        elif shape_kind in {"capsule", "pill"}:
            style["rounding_in"] = min(min(width, height) / 2, float(node.get("max_rounding_in", 0.45)))
        else:
            style["rounding_in"] = min(width, height) * 0.18

    shape = draw_rectangle(page, page_height, node)
    apply_style(shape, style)

    text = node.get("text")
    if text:
        x = float(node["x"])
        y = float(node["y"])
        title_h = float(node.get("title_h_in", min(0.24, max(0.14, float(node["h"]) * 0.10))))
        title_x = x + float(node.get("title_pad_x_in", 0.08))
        title_y = y + float(node.get("title_pad_y_in", 0.02))
        title_w = max(0.1, width - float(node.get("title_pad_x_in", 0.08)) * 2)
        title_style = merge_style(
            style,
            {
                "fill": "none",
                "line": "none",
                "font_size_pt": node.get("title_font_size_pt", style.get("font_size_pt", 15)),
                "text_align": node.get("title_align", 0),
            },
        )
        draw_text_box(page, page_height, title_x, title_y, title_w, title_h, str(text), title_style)
    return shape


def branch_offsets(node: dict[str, Any], style: dict[str, Any], total: float) -> list[float]:
    raw_positions = node.get("branch_positions", node.get("positions"))
    if isinstance(raw_positions, list) and raw_positions:
        offsets: list[float] = []
        for value in raw_positions:
            if not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            offsets.append(numeric * total if 0.0 <= numeric <= 1.0 else numeric)
        if offsets:
            return offsets

    count = max(1, int(node.get("branch_count", style.get("branch_count", 4))))
    if count == 1:
        return [total / 2]
    return [total * index / (count - 1) for index in range(count)]


def draw_boundary_fanout(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    side = str(node.get("side", "right")).lower()
    line_style = merge_style(style, {"fill": "none", "end_arrow": style.get("end_arrow", "triangle")})
    branch_labels = [str(item) for item in node.get("labels", [])] if isinstance(node.get("labels"), list) else []
    label_gap = float(node.get("label_gap_in", style.get("label_gap_in", 0.04)))
    label_w = float(node.get("label_width_in", style.get("label_width_in", 0.32)))
    label_h = float(node.get("label_height_in", style.get("label_height_in", 0.18)))
    label_style = merge_style(style, {"fill": "none", "line": "none", "font_size_pt": node.get("label_font_size_pt", 11)})

    shape = None
    if side in {"right", "left"}:
        for index, offset in enumerate(branch_offsets(node, style, height)):
            line_y = y + offset
            if side == "right":
                start = (x, line_y)
                end = (x + width, line_y)
                label_x = end[0] + label_gap
            else:
                start = (x + width, line_y)
                end = (x, line_y)
                label_x = end[0] - label_gap - label_w
            shape = draw_line_segment(page, page_height, start, end, line_style)
            if index < len(branch_labels):
                draw_text_box(page, page_height, label_x, line_y - label_h / 2, label_w, label_h, branch_labels[index], label_style)
    elif side in {"top", "bottom"}:
        for index, offset in enumerate(branch_offsets(node, style, width)):
            line_x = x + offset
            if side == "bottom":
                start = (line_x, y)
                end = (line_x, y + height)
                label_y = end[1] + label_gap
            else:
                start = (line_x, y + height)
                end = (line_x, y)
                label_y = end[1] - label_gap - label_h
            shape = draw_line_segment(page, page_height, start, end, line_style)
            if index < len(branch_labels):
                draw_text_box(page, page_height, line_x - label_w / 2, label_y, label_w, label_h, branch_labels[index], label_style)
    else:
        raise ValueError(f"Unsupported boundary_fanout side: {side}")
    return shape


def draw_rotated_diamond(page: Any, page_height: float, node: dict[str, Any]) -> Any:
    width = float(node["w"]) / math.sqrt(2)
    height = float(node["h"]) / math.sqrt(2)
    cx = float(node["x"]) + float(node["w"]) / 2
    cy = to_visio_y(page_height, float(node["y"]) + float(node["h"]) / 2)
    shape = page.DrawRectangle(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    try_set_formula(shape, "Angle", "45 deg")
    try_set_formula(shape, "TxtAngle", "-45 deg")
    return shape


def draw_image_tile(page: Any, page_height: float, node: dict[str, Any], asset_path: Path) -> Any:
    shape = page.Import(str(asset_path))
    cx = float(node["x"]) + float(node["w"]) / 2
    cy = to_visio_y(page_height, float(node["y"]) + float(node["h"]) / 2)
    try_set_result(shape, "PinX", cx)
    try_set_result(shape, "PinY", cy)
    try_set_result(shape, "Width", float(node["w"]))
    try_set_result(shape, "Height", float(node["h"]))
    return shape


def draw_wave_signal(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    baseline = y + height * float(node.get("baseline_ratio", 0.5))
    amplitude = float(node.get("amplitude_in", height * float(style.get("amplitude_ratio", 0.38))))
    samples = node.get("samples")

    values: list[float] = []
    if isinstance(samples, list) and samples:
        for item in samples:
            if isinstance(item, (int, float)):
                values.append(float(item))
    if not values:
        point_count = max(8, int(node.get("point_count", 48)))
        cycles = float(node.get("cycles", style.get("cycles", 2.5)))
        values = [
            math.sin(2 * math.pi * cycles * index / (point_count - 1))
            for index in range(point_count)
        ]

    if len(values) == 1:
        values = [values[0], values[0]]

    line_style = merge_style(style, {"fill": "none", "end_arrow": "none"})
    shape = None
    if node.get("show_baseline"):
        shape = draw_line_segment(page, page_height, (x, baseline), (x + width, baseline), line_style)

    points = []
    for index, value in enumerate(values):
        px = x + width * index / (len(values) - 1)
        py = baseline - max(-1.0, min(1.0, float(value))) * amplitude
        points.append((px, py))

    for start, end in zip(points, points[1:]):
        shape = draw_line_segment(page, page_height, start, end, line_style)
    return shape


def classifier_blocks(node: dict[str, Any]) -> list[dict[str, Any]]:
    raw_blocks = node.get("blocks", node.get("labels", ["AvgPool", "Linear"]))
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raw_blocks = ["AvgPool", "Linear"]

    blocks: list[dict[str, Any]] = []
    for item in raw_blocks:
        if isinstance(item, dict):
            blocks.append(item)
        else:
            blocks.append({"text": str(item)})
    return blocks


def draw_classifier_head(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    orientation = str(node.get("orientation", "horizontal")).lower()
    if orientation in {"vertical", "v"}:
        return draw_classifier_head_vertical(page, page_height, node, style)
    if orientation not in {"horizontal", "h"}:
        raise ValueError("classifier_head orientation must be horizontal or vertical.")

    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    padding = float(node.get("padding_in", style.get("padding_in", 0.04)))
    gap = float(node.get("block_gap_in", style.get("block_gap_in", 0.08)))
    inner_w = max(0.01, width - padding * 2)
    inner_h = max(0.01, height - padding * 2)
    blocks = classifier_blocks(node)
    output_labels = [str(item) for item in node.get("output_labels", [])] if isinstance(node.get("output_labels"), list) else []
    fanout_count = int(node.get("fanout_count", len(output_labels) or 0))
    if output_labels:
        fanout_count = max(fanout_count, len(output_labels))
    output_mode = str(node.get("output_mode", "internal_fanout" if fanout_count else "none")).lower()
    if output_mode in {"none", "boundary", "boundary_fanout", "container_boundary", "external"}:
        fanout_count = 0

    label_w = min(0.42, max(0.18, width * 0.12)) if output_labels else 0.0
    fan_zone_w = min(inner_w * 0.24, max(0.22, float(node.get("fanout_width_in", inner_w * 0.16)))) if fanout_count else 0.0
    block_area_w = max(0.01, inner_w - fan_zone_w - (gap if fanout_count else 0.0) - label_w)
    block_w = max(0.01, (block_area_w - gap * (len(blocks) - 1)) / len(blocks))
    block_h = min(inner_h, float(node.get("block_height_in", inner_h * 0.56)))
    block_y = y + padding + (inner_h - block_h) / 2
    block_style = merge_style(style, {"fill": style.get("fill", "#FFFFFF"), "line": style.get("line", "#111827")})
    connector_style = merge_style(
        {
            "line": style.get("line", "#111827"),
            "line_weight_pt": style.get("line_weight_pt", 1.0),
            "line_dash": style.get("line_dash", "solid"),
            "end_arrow": "triangle",
        },
        node.get("connector_style") if isinstance(node.get("connector_style"), dict) else None,
    )

    shape = None
    previous_center: tuple[float, float] | None = None
    last_right = x + padding
    last_center_y = y + height / 2
    for index, block in enumerate(blocks):
        block_x = x + padding + index * (block_w + gap)
        block_node = {"x": block_x, "y": block_y, "w": block_w, "h": block_h}
        shape = draw_rectangle(page, page_height, block_node)
        apply_style(shape, merge_style(block_style, block.get("style") if isinstance(block.get("style"), dict) else None))
        text = str(block.get("text", block.get("label", "")))
        if text:
            try_set_text(shape, text)

        center = (block_x + block_w / 2, block_y + block_h / 2)
        if previous_center is not None:
            draw_line_segment(
                page,
                page_height,
                (previous_center[0] + block_w / 2, previous_center[1]),
                (block_x, center[1]),
                connector_style,
            )
        previous_center = center
        last_right = block_x + block_w
        last_center_y = center[1]

    if fanout_count:
        trunk_x = min(x + width - padding - label_w - 0.12, last_right + gap + fan_zone_w * 0.35)
        trunk_x = max(trunk_x, last_right + gap)
        fan_end_x = x + width - padding - label_w
        fan_top = y + padding
        fan_bottom = y + height - padding
        if fanout_count == 1:
            branch_ys = [last_center_y]
        else:
            branch_ys = [
                fan_top + (fan_bottom - fan_top) * index / (fanout_count - 1)
                for index in range(fanout_count)
            ]

        draw_line_segment(page, page_height, (last_right, last_center_y), (trunk_x, last_center_y), merge_style(connector_style, {"end_arrow": "none"}))
        if len(branch_ys) > 1:
            draw_line_segment(page, page_height, (trunk_x, min(branch_ys)), (trunk_x, max(branch_ys)), merge_style(connector_style, {"end_arrow": "none"}))
        for index, branch_y in enumerate(branch_ys):
            shape = draw_line_segment(page, page_height, (trunk_x, branch_y), (fan_end_x, branch_y), connector_style)
            if index < len(output_labels):
                draw_text_box(
                    page,
                    page_height,
                    fan_end_x + 0.02,
                    branch_y - min(0.12, height / 8),
                    max(0.12, label_w - 0.02),
                    min(0.24, height / 4),
                    output_labels[index],
                    merge_style(style, {"font_size_pt": max(6, float(style.get("font_size_pt", 10)) - 1)}),
                )

    return shape


def draw_classifier_head_vertical(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    padding = float(node.get("padding_in", style.get("padding_in", 0.04)))
    gap = float(node.get("block_gap_in", style.get("vertical_block_gap_in", style.get("block_gap_in", 0.14))))
    inner_w = max(0.01, width - padding * 2)
    inner_h = max(0.01, height - padding * 2)
    blocks = classifier_blocks(node)
    block_w = min(inner_w, float(node.get("block_width_in", inner_w * 0.88)))
    requested_block_h = node.get("block_height_in")
    if requested_block_h is not None:
        block_h = min(inner_h, float(requested_block_h))
    else:
        block_h = max(0.01, (inner_h - gap * (len(blocks) - 1)) / len(blocks))
    block_x = x + padding + (inner_w - block_w) / 2
    total_h = block_h * len(blocks) + gap * (len(blocks) - 1)
    block_y = y + padding + max(0.0, (inner_h - total_h) / 2)
    block_style = merge_style(style, {"fill": style.get("fill", "#FFFFFF"), "line": style.get("line", "#111827")})
    connector_style = merge_style(
        {
            "line": style.get("line", "#111827"),
            "line_weight_pt": style.get("line_weight_pt", 1.0),
            "line_dash": style.get("line_dash", "solid"),
            "end_arrow": "triangle",
            "arrow_size": node.get("internal_arrow_size", style.get("internal_arrow_size", "small")),
        },
        node.get("connector_style") if isinstance(node.get("connector_style"), dict) else None,
    )

    shape = None
    previous_bottom: tuple[float, float] | None = None
    for index, block in enumerate(blocks):
        current_y = block_y + index * (block_h + gap)
        block_node = {"x": block_x, "y": current_y, "w": block_w, "h": block_h}
        shape = draw_rectangle(page, page_height, block_node)
        apply_style(shape, merge_style(block_style, block.get("style") if isinstance(block.get("style"), dict) else None))
        text = str(block.get("text", block.get("label", "")))
        if text:
            try_set_text(shape, text)

        current_top = (block_x + block_w / 2, current_y)
        if previous_bottom is not None:
            draw_line_segment(page, page_height, previous_bottom, current_top, connector_style)
        previous_bottom = (block_x + block_w / 2, current_y + block_h)

    return shape


def matrix_cell_styles(node: dict[str, Any], style: dict[str, Any]) -> dict[tuple[int, int], str]:
    index_base = int(node.get("index_base", 0))
    default_fill = str(style.get("active_fill", "#2B7C8E"))
    cells: dict[tuple[int, int], str] = {}

    for item in node.get("colored_cells", node.get("cells", [])):
        if isinstance(item, dict):
            row = int(item["row"]) - index_base
            col = int(item["col"]) - index_base
            fill = str(item.get("fill", default_fill))
        elif isinstance(item, list) and len(item) >= 2:
            row = int(item[0]) - index_base
            col = int(item[1]) - index_base
            fill = str(item[2]) if len(item) >= 3 else default_fill
        else:
            continue
        cells[(row, col)] = fill
    return cells


def draw_grid_matrix(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    rows = int(node["rows"])
    cols = int(node["cols"])
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    cell_w = width / cols
    cell_h = height / rows
    base_fill = str(style.get("cell_fill", style.get("fill", "#FFFFFF")))
    colored = matrix_cell_styles(node, style)
    first_shape = None

    for row in range(rows):
        for col in range(cols):
            cell_node = {
                "x": x + col * cell_w,
                "y": y + row * cell_h,
                "w": cell_w,
                "h": cell_h,
            }
            cell_shape = draw_rectangle(page, page_height, cell_node)
            apply_style(
                cell_shape,
                {
                    "fill": colored.get((row, col), base_fill),
                    "line": "none",
                },
            )
            if first_shape is None:
                first_shape = cell_shape

    grid_line_style = {
        "line": style.get("grid_line", style.get("line", "#000000")),
        "line_weight_pt": style.get("grid_line_weight_pt", style.get("line_weight_pt", 1.0)),
        "line_dash": style.get("line_dash", "solid"),
        "end_arrow": "none",
    }

    for col in range(cols + 1):
        gx = x + col * cell_w
        first_shape = draw_line_segment(page, page_height, (gx, y), (gx, y + height), grid_line_style)

    for row in range(rows + 1):
        gy = y + row * cell_h
        first_shape = draw_line_segment(page, page_height, (x, gy), (x + width, gy), grid_line_style)

    return first_shape


def draw_stacked_process(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    layers = max(1, int(node.get("layers", style.get("layers", 4))))
    dx = float(node.get("stack_dx_in", style.get("stack_dx_in", -0.04)))
    dy = float(node.get("stack_dy_in", style.get("stack_dy_in", 0.035)))
    shape = None

    for index in reversed(range(layers)):
        layer_node = dict(node)
        layer_node["x"] = float(node["x"]) + dx * index
        layer_node["y"] = float(node["y"]) + dy * index
        shape = draw_rectangle(page, page_height, layer_node)
        apply_style(shape, style)

    return shape


def relative_or_absolute(value: Any, total: float) -> float:
    numeric = float(value)
    if -1.0 <= numeric <= 1.0:
        return numeric * total
    return numeric


def draw_notched_block(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    base = draw_rectangle(page, page_height, node)
    apply_style(base, style)

    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    notches = node.get("notches") or [
        {"x": 0.50, "y": 0.28, "w": 0.38, "h": 0.16, "shape": "diamond"},
        {"x": 0.50, "y": 0.72, "w": 0.38, "h": 0.16, "shape": "diamond"},
    ]
    notch_fill = str(node.get("notch_fill", style.get("notch_fill", "#FFFFFF")))

    for index, notch in enumerate(notches):
        nw = relative_or_absolute(notch.get("w", 0.25), width)
        nh = relative_or_absolute(notch.get("h", 0.15), height)
        nx = x + relative_or_absolute(notch.get("x", 0.5), width) - nw / 2
        ny = y + relative_or_absolute(notch.get("y", 0.5), height) - nh / 2
        notch_node = {"x": nx, "y": ny, "w": nw, "h": nh}
        if str(notch.get("shape", "diamond")).lower() == "rectangle":
            shape = draw_rectangle(page, page_height, notch_node)
        else:
            shape = draw_rotated_diamond(page, page_height, notch_node)
        apply_style(
            shape,
            {
                "fill": str(notch.get("fill", notch_fill)),
                "line": str(notch.get("line", notch.get("fill", notch_fill))),
                "line_weight_pt": float(notch.get("line_weight_pt", 0)),
            },
        )

    return base


def draw_feature_map_banded(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    orientation = str(node.get("orientation", "horizontal")).lower()
    bands = node.get("bands") or node.get("stripe_colors") or [
        "#B7DCEB",
        "#F8E49B",
        "#B7DCEB",
        "#C9C0D8",
        "#D7D7D7",
    ]

    parsed_bands: list[dict[str, Any]] = []
    for band in bands:
        if isinstance(band, dict):
            parsed_bands.append(band)
        else:
            parsed_bands.append({"fill": str(band), "size": 1})
    total_size = sum(float(band.get("size", 1)) for band in parsed_bands) or 1.0
    cursor = 0.0
    first_shape = None
    for band in parsed_bands:
        ratio = float(band.get("size", 1)) / total_size
        if orientation == "vertical":
            band_node = {"x": x + cursor * width, "y": y, "w": width * ratio, "h": height}
        else:
            band_node = {"x": x, "y": y + cursor * height, "w": width, "h": height * ratio}
        shape = draw_rectangle(page, page_height, band_node)
        apply_style(shape, {"fill": str(band.get("fill", "#FFFFFF")), "line": "none"})
        first_shape = first_shape or shape
        cursor += ratio

    for overlay in node.get("overlays", node.get("vertical_bands", [])):
        ox = x + relative_or_absolute(overlay.get("x", 0), width)
        oy = y + relative_or_absolute(overlay.get("y", 0), height)
        ow = relative_or_absolute(overlay.get("w", width), width)
        oh = relative_or_absolute(overlay.get("h", height), height)
        shape = draw_rectangle(page, page_height, {"x": ox, "y": oy, "w": ow, "h": oh})
        apply_style(
            shape,
            {
                "fill": str(overlay.get("fill", "#000000")),
                "line": str(overlay.get("line", overlay.get("fill", "#000000"))),
                "line_weight_pt": float(overlay.get("line_weight_pt", 0)),
            },
        )
        first_shape = first_shape or shape

    outline = draw_rectangle(page, page_height, node)
    apply_style(outline, merge_style(style, {"fill": "none"}))
    return outline or first_shape


def sequence_from_bands(raw_values: Any, default_values: list[str]) -> list[str]:
    if not isinstance(raw_values, list) or not raw_values:
        return default_values
    values: list[str] = []
    for item in raw_values:
        if isinstance(item, dict):
            values.append(str(item.get("fill", item.get("color", "#FFFFFF"))))
        else:
            values.append(str(item))
    return values or default_values


def numeric_sequence(raw_values: Any, count: int, default_value: float = 0.0) -> list[float]:
    if isinstance(raw_values, list) and raw_values:
        parsed = [float(item) for item in raw_values if isinstance(item, (int, float))]
        if parsed:
            if len(parsed) >= count:
                return parsed[:count]
            return [parsed[index % len(parsed)] for index in range(count)]
    return [default_value for _ in range(count)]


def normalized_weights(raw_values: Any, count: int) -> list[float]:
    if isinstance(raw_values, list) and raw_values:
        parsed = [float(item) for item in raw_values if isinstance(item, (int, float)) and float(item) > 0]
        if parsed:
            if len(parsed) < count:
                parsed.extend([parsed[-1]] * (count - len(parsed)))
            weights = parsed[:count]
            total = sum(weights) or 1.0
            return [value / total for value in weights]
    return [1.0 / count for _ in range(count)]


def draw_feature_map_grid(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    row_palette = sequence_from_bands(
        node.get("row_colors", node.get("bands", style.get("row_colors"))),
        ["#F2A66F", "#A8D7E5", "#C8D9C2", "#F3E889", "#9BC6D9", "#F2A66F"],
    )
    rows = int(node.get("rows", len(row_palette)))
    cols = int(node.get("cols", node.get("columns", 8)))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"feature_map_grid `{node.get('id', '<unknown>')}` needs positive rows/cols.")

    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    row_weights = normalized_weights(node.get("row_weights", node.get("row_heights")), rows)
    col_weights = normalized_weights(node.get("column_weights", node.get("column_widths")), cols)
    column_shades = numeric_sequence(node.get("column_shades", style.get("column_shades")), cols, 0.0)
    shade_color = str(node.get("shade_color", style.get("shade_color", "#111111")))
    max_shade = float(node.get("max_shade", style.get("max_shade", 0.58)))

    first_shape = None
    row_tops: list[float] = [y]
    cursor_y = y
    for row_weight in row_weights:
        cursor_y += height * row_weight
        row_tops.append(cursor_y)
    col_lefts: list[float] = [x]
    cursor_x = x
    for col_weight in col_weights:
        cursor_x += width * col_weight
        col_lefts.append(cursor_x)

    for row in range(rows):
        base_fill = row_palette[row % len(row_palette)]
        for col in range(cols):
            shade_amount = max(0.0, min(1.0, column_shades[col])) * max_shade
            fill = blend_hex_colors(base_fill, shade_color, shade_amount) if shade_amount else base_fill
            cell_node = {
                "x": col_lefts[col],
                "y": row_tops[row],
                "w": col_lefts[col + 1] - col_lefts[col],
                "h": row_tops[row + 1] - row_tops[row],
            }
            cell_shape = draw_rectangle(page, page_height, cell_node)
            apply_style(
                cell_shape,
                {
                    "fill": fill,
                    "line": "none",
                    "fill_transparency_pct": style.get("fill_transparency_pct", 0),
                },
            )
            first_shape = first_shape or cell_shape

    separator_style = {
        "line": node.get("grid_line", style.get("grid_line", "#333333")),
        "line_weight_pt": node.get("grid_line_weight_pt", style.get("grid_line_weight_pt", 0.35)),
        "line_transparency_pct": node.get("grid_line_transparency_pct", style.get("grid_line_transparency_pct", 35)),
        "end_arrow": "none",
    }
    if node.get("show_column_lines", True):
        for col in range(1, cols):
            first_shape = draw_line_segment(page, page_height, (col_lefts[col], y), (col_lefts[col], y + height), separator_style)
    if node.get("show_row_lines", False):
        for row in range(1, rows):
            first_shape = draw_line_segment(page, page_height, (x, row_tops[row]), (x + width, row_tops[row]), separator_style)

    outline = draw_rectangle(page, page_height, node)
    apply_style(
        outline,
        {
            "fill": "none",
            "line": node.get("outline", style.get("outline", style.get("line", "#111111"))),
            "line_weight_pt": node.get("outline_weight_pt", style.get("outline_weight_pt", 0.9)),
            "line_dash": style.get("line_dash", "solid"),
        },
    )
    return outline or first_shape


def draw_merge_bus(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    orientation = str(node.get("orientation", "vertical")).lower()
    port_positions = [float(item) for item in node.get("port_positions", [0, 0.5, 1])]
    side = str(node.get("side", "left")).lower()
    port_length = float(node.get("port_length_in", style.get("port_length_in", 0.18)))
    line_style = merge_style(style, {"fill": "none", "end_arrow": "none"})

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if orientation == "horizontal":
        spine_y = y + height / 2
        segments.append(((x, spine_y), (x + width, spine_y)))
        for pos in port_positions:
            px = x + width * max(0.0, min(1.0, pos))
            if side in {"top", "both"}:
                segments.append(((px, spine_y), (px, spine_y - port_length)))
            if side in {"bottom", "both"}:
                segments.append(((px, spine_y), (px, spine_y + port_length)))
    else:
        spine_x = x + width / 2
        segments.append(((spine_x, y), (spine_x, y + height)))
        for pos in port_positions:
            py = y + height * max(0.0, min(1.0, pos))
            if side in {"left", "both"}:
                segments.append(((spine_x, py), (spine_x - port_length, py)))
            if side in {"right", "both"}:
                segments.append(((spine_x, py), (spine_x + port_length, py)))

    shape = None
    for start, end in segments:
        shape = draw_line_segment(page, page_height, start, end, line_style)
    return shape


def draw_bracket(page: Any, page_height: float, node: dict[str, Any], style: dict[str, Any]) -> Any:
    x = float(node["x"])
    y = float(node["y"])
    width = float(node["w"])
    height = float(node["h"])
    orientation = str(node.get("orientation", "right")).lower()
    line_style = merge_style(style, {"fill": "none", "end_arrow": "none"})
    ticks = node.get("tick_positions")
    if ticks is None:
        ticks = [0, 0.5, 1] if node.get("middle_tick") else [0, 1]
    tick_positions = [max(0.0, min(1.0, float(tick))) for tick in ticks]

    if orientation == "right":
        spine_x = x + width
        segments = [((spine_x, y), (spine_x, y + height))]
        segments.extend(((x, y + height * tick), (spine_x, y + height * tick)) for tick in tick_positions)
    elif orientation == "left":
        spine_x = x
        segments = [((spine_x, y), (spine_x, y + height))]
        segments.extend(((spine_x, y + height * tick), (x + width, y + height * tick)) for tick in tick_positions)
    elif orientation == "down":
        spine_y = y + height
        segments = [((x, spine_y), (x + width, spine_y))]
        segments.extend(((x + width * tick, y), (x + width * tick, spine_y)) for tick in tick_positions)
    elif orientation == "up":
        spine_y = y
        segments = [((x, spine_y), (x + width, spine_y))]
        segments.extend(((x + width * tick, spine_y), (x + width * tick, y + height)) for tick in tick_positions)
    else:
        raise ValueError(f"Unsupported bracket orientation: {orientation}")

    shape = None
    for start, end in segments:
        shape = draw_line_segment(page, page_height, start, end, line_style)
    return shape


def candidate_stencil_paths() -> list[Path]:
    candidates: list[Path] = []
    roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        r"C:\Program Files\Microsoft Office",
    ]
    locales = ["2052", "1033"]
    for root in [item for item in roots if item]:
        for locale in locales:
            candidates.append(
                Path(root) / "Microsoft Office" / "root" / "Office16" / "Visio Content" / locale / "BASFLO_M.VSSX"
            )
            candidates.append(
                Path(root) / "root" / "Office16" / "Visio Content" / locale / "BASFLO_M.VSSX"
            )
    return candidates


def open_basic_flow_stencil(app: Any) -> Any | None:
    for path in candidate_stencil_paths():
        if path.exists():
            try:
                return app.Documents.OpenEx(str(path), 64)
            except Exception:
                continue
    return None


def get_master(stencil: Any | None, names: list[str]) -> Any | None:
    if stencil is None:
        return None
    for name in names:
        for getter in ("ItemU", "Item"):
            try:
                return getattr(stencil.Masters, getter)(name)
            except Exception:
                continue
    return None


def draw_master_shape(
    page: Any,
    page_height: float,
    node: dict[str, Any],
    master: Any | None,
) -> Any | None:
    if master is None:
        return None
    cx = float(node["x"]) + float(node["w"]) / 2
    cy = to_visio_y(page_height, float(node["y"]) + float(node["h"]) / 2)
    try:
        shape = page.Drop(master, cx, cy)
    except Exception:
        return None
    try_set_result(shape, "Width", float(node["w"]))
    try_set_result(shape, "Height", float(node["h"]))
    return shape


def node_style(
    node: dict[str, Any],
    component_map: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    node_type = node["type"]
    definition = component_map["node_types"][node_type]
    profile_nodes = profile.get("node_types", {})
    style = merge_style(
        definition.get("default_style"),
        profile.get("global_text"),
        profile_nodes.get(node_type),
        node.get("style"),
    )
    return definition["renderer"], style, definition


def draw_node(
    page: Any,
    page_height: float,
    node: dict[str, Any],
    asset_paths: dict[str, Path],
    component_map: dict[str, Any],
    profile: dict[str, Any],
    masters: dict[str, Any | None],
) -> Any:
    renderer, style, definition = node_style(node, component_map, profile)
    shape = None
    render_kind = renderer

    if renderer == "visio_master":
        shape = draw_master_shape(page, page_height, node, masters.get(definition.get("master")))
        if shape is None:
            render_kind = definition.get("fallback_renderer", "rectangle")

    if shape is None:
        if render_kind == "group_container":
            shape = draw_group_container(page, page_height, node, style)
        elif render_kind == "audit_region":
            shape = draw_rectangle(page, page_height, node)
        elif render_kind in {"rectangle", "rounded_rectangle", "terminator", "pill", "legend_block", "text_block"}:
            shape = draw_rectangle(page, page_height, node)
        elif render_kind == "oval":
            shape = draw_oval(page, page_height, node)
        elif render_kind == "operator_node":
            shape = draw_operator_node(page, page_height, node, style)
        elif render_kind == "diamond":
            shape = draw_rotated_diamond(page, page_height, node)
        elif render_kind == "bracket":
            shape = draw_bracket(page, page_height, node, style)
        elif render_kind == "junction_point":
            shape = draw_oval(page, page_height, node)
        elif render_kind == "boundary_port":
            shape = draw_boundary_port(page, page_height, node, style)
        elif render_kind == "image_tile":
            asset_ref = node.get("asset_ref")
            if not asset_ref or asset_ref not in asset_paths:
                raise ValueError(f"image_tile node `{node['id']}` requires a valid `asset_ref`.")
            shape = draw_image_tile(page, page_height, node, asset_paths[asset_ref])
        elif render_kind == "grid_matrix":
            shape = draw_grid_matrix(page, page_height, node, style)
        elif render_kind == "stacked_process":
            shape = draw_stacked_process(page, page_height, node, style)
        elif render_kind == "notched_block":
            shape = draw_notched_block(page, page_height, node, style)
        elif render_kind == "feature_map_banded":
            shape = draw_feature_map_banded(page, page_height, node, style)
        elif render_kind == "feature_map_grid":
            shape = draw_feature_map_grid(page, page_height, node, style)
        elif render_kind == "merge_bus":
            shape = draw_merge_bus(page, page_height, node, style)
        elif render_kind == "wave_signal":
            shape = draw_wave_signal(page, page_height, node, style)
        elif render_kind == "classifier_head":
            shape = draw_classifier_head(page, page_height, node, style)
        elif render_kind == "boundary_fanout":
            shape = draw_boundary_fanout(page, page_height, node, style)
        else:
            raise ValueError(f"Unsupported renderer: {render_kind}")

    text = node.get("text")
    if text and render_kind not in {"bracket", "wave_signal", "classifier_head", "boundary_fanout", "feature_map_grid", "group_container", "audit_region", "operator_node"}:
        try_set_text(shape, str(text))

    if render_kind not in {"bracket", "feature_map_banded", "feature_map_grid", "merge_bus", "wave_signal", "classifier_head", "boundary_fanout", "group_container", "operator_node"}:
        apply_style(shape, style)
    if render_kind == "terminator" and "rounding_in" not in style:
        try_set_formula(shape, "Rounding", f"{min(float(node['h']) / 2, 0.25)} in")
    return shape


def opposite_axis(side_a: str, side_b: str) -> str | None:
    if {side_a, side_b} <= {"left", "right"}:
        return "horizontal"
    if {side_a, side_b} <= {"top", "bottom"}:
        return "vertical"
    return None


def orthogonal_points(
    start: tuple[float, float],
    end: tuple[float, float],
    axis: str | None,
) -> list[tuple[float, float]]:
    sx, sy = start
    tx, ty = end
    if axis == "horizontal":
        mid_x = (sx + tx) / 2
        return [start, (mid_x, sy), (mid_x, ty), end]
    if axis == "vertical":
        mid_y = (sy + ty) / 2
        return [start, (sx, mid_y), (tx, mid_y), end]
    if abs(tx - sx) >= abs(ty - sy):
        mid_x = (sx + tx) / 2
        return [start, (mid_x, sy), (mid_x, ty), end]
    mid_y = (sy + ty) / 2
    return [start, (sx, mid_y), (tx, mid_y), end]


def snap_axis_segments(
    points: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    if len(points) < 2 or tolerance <= 0:
        return points

    snapped = [points[0]]
    for x, y in points[1:]:
        prev_x, prev_y = snapped[-1]
        if abs(x - prev_x) <= tolerance:
            x = prev_x
        if abs(y - prev_y) <= tolerance:
            y = prev_y
        snapped.append((x, y))
    return snapped


def edge_route_points(
    edge: dict[str, Any],
    style: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[tuple[float, float]]:
    from_ref = edge.get("from")
    to_ref = edge.get("to")
    from_point = edge_point(edge, "from")
    to_point = edge_point(edge, "to")

    from_node = nodes_by_id[node_id_from_endpoint(from_ref)] if isinstance(from_ref, str) else None
    to_node = nodes_by_id[node_id_from_endpoint(to_ref)] if isinstance(to_ref, str) else None
    from_peer_point = to_point or (node_center_point(to_node) if to_node else from_point)
    to_peer_point = from_point or (node_center_point(from_node) if from_node else to_point)
    if from_peer_point is None or to_peer_point is None:
        raise ValueError(f"Edge `{edge.get('id', '<unknown>')}` requires resolvable endpoints.")

    from_side = side_of(from_ref, from_node, fake_node_at(from_peer_point)) if isinstance(from_ref, str) and from_node else "point"
    to_side = side_of(to_ref, to_node, fake_node_at(to_peer_point)) if isinstance(to_ref, str) and to_node else "point"
    start = resolve_edge_endpoint(edge, "from", from_peer_point, nodes_by_id)
    end = resolve_edge_endpoint(edge, "to", to_peer_point, nodes_by_id)

    explicit_points = edge.get("points") or []
    axis_snap = float(edge.get("axis_snap_in", style.get("axis_snap_in", 0.03)))
    if explicit_points:
        return snap_axis_segments(
            [start, *[(float(x), float(y)) for x, y in explicit_points], end],
            axis_snap,
        )

    route = edge.get("route") or style.get("route") or "auto"
    if route == "straight":
        return [start, end]
    if route in {"horizontal", "hline", "axis_horizontal"}:
        return [(start[0], start[1]), (end[0], start[1])]
    if route in {"vertical", "vline", "axis_vertical"}:
        return [(start[0], start[1]), (start[0], end[1])]
    if route in {"hv", "horizontal_then_vertical"}:
        return snap_axis_segments([start, (end[0], start[1]), end], axis_snap)
    if route in {"vh", "vertical_then_horizontal"}:
        return snap_axis_segments([start, (start[0], end[1]), end], axis_snap)

    axis = opposite_axis(from_side, to_side)
    snap_tolerance = float(edge.get("snap_tolerance_in", style.get("snap_tolerance_in", 0.18)))

    if route in {"orthogonal", "elbow", "right_angle"}:
        return snap_axis_segments(orthogonal_points(start, end, axis), axis_snap)

    if axis == "horizontal" and abs(start[1] - end[1]) <= snap_tolerance:
        y = (start[1] + end[1]) / 2
        return [(start[0], y), (end[0], y)]
    if axis == "vertical" and abs(start[0] - end[0]) <= snap_tolerance:
        x = (start[0] + end[0]) / 2
        return [(x, start[1]), (x, end[1])]
    if axis:
        return snap_axis_segments(orthogonal_points(start, end, axis), axis_snap)
    return [start, end]


def draw_line_segment(
    page: Any,
    page_height: float,
    start: tuple[float, float],
    end: tuple[float, float],
    style: dict[str, Any],
) -> Any:
    segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
    shape = page.DrawLine(start[0], to_visio_y(page_height, start[1]), end[0], to_visio_y(page_height, end[1]))
    apply_style(shape, style)
    if style.get("end_arrow") == "triangle":
        try_set_result(shape, "EndArrow", 13)
        try_set_result(shape, "EndArrowSize", arrow_size_value(style.get("arrow_size", style.get("end_arrow_size")), segment_length))
    elif style.get("end_arrow") == "none":
        try_set_result(shape, "EndArrow", 0)
    if style.get("begin_arrow") == "triangle":
        try_set_result(shape, "BeginArrow", 13)
        try_set_result(shape, "BeginArrowSize", arrow_size_value(style.get("begin_arrow_size", style.get("arrow_size")), segment_length))
    return shape


def edge_style(edge: dict[str, Any], component_map: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    definition = component_map["edge_types"][edge["type"]]
    profile_edges = profile.get("edge_types", {})
    return merge_style(
        definition.get("default_style"),
        profile.get("global_edge"),
        profile_edges.get(edge["type"]),
        edge.get("style"),
    )


def draw_edge(
    page: Any,
    page_height: float,
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    component_map: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[Any, float, float]:
    style = edge_style(edge, component_map, profile)
    points = edge_route_points(edge, style, nodes_by_id)
    segments = []
    for index in range(len(points) - 1):
        segment_style = dict(style)
        if index != len(points) - 2:
            segment_style["end_arrow"] = "none"
        segments.append(draw_line_segment(page, page_height, points[index], points[index + 1], segment_style))

    mid_index = len(points) // 2
    mid_x, mid_y = points[mid_index]
    return segments[-1], mid_x, mid_y


def draw_edge_label(page: Any, page_height: float, text: str, mid_x: float, mid_y: float, profile: dict[str, Any]) -> Any:
    shape = page.DrawRectangle(
        mid_x - 0.45,
        to_visio_y(page_height, mid_y + 0.12),
        mid_x + 0.45,
        to_visio_y(page_height, mid_y - 0.12),
    )
    try_set_text(shape, text)
    apply_style(
        shape,
        merge_style(
            profile.get("global_text"),
            {
                "fill": "none",
                "line": "none",
                "font_size_pt": 10,
                "font_weight": "regular",
            },
        ),
    )
    return shape


def resolve_profile(scene: dict[str, Any], profiles: dict[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    name = (
        requested
        or scene.get("metadata", {}).get("style_profile")
        or scene.get("page", {}).get("style_profile")
        or profiles.get("default_profile")
        or "paper_white"
    )
    profile = profiles.get("profiles", {}).get(name, {})
    return name, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a visiomaster scene.json into Visio.")
    parser.add_argument("scene", help="Path to scene.json")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered outputs")
    parser.add_argument("--visible", action="store_true", help="Show Visio while rendering")
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave Visio open after rendering. Implies --visible.",
    )
    parser.add_argument("--basename", help="Optional output basename")
    parser.add_argument("--style-profile", help="Override scene style profile.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = normalize_scene_coordinates(load_json(scene_path))
    component_map = load_component_map()
    profiles = load_style_profiles()
    profile_name, profile = resolve_profile(scene, profiles, args.style_profile)

    try:
        import win32com.client.gencache as gencache
    except ImportError as exc:
        raise SystemExit("pywin32 is required. Install it in the active Python environment.") from exc

    app = gencache.EnsureDispatch("Visio.Application")
    app.Visible = bool(args.visible or args.keep_open)
    try:
        app.AlertResponse = 7
    except Exception:
        pass

    stencil = open_basic_flow_stencil(app)
    masters = {
        "Process": get_master(stencil, ["Process", "流程"]),
        "Decision": get_master(stencil, ["Decision", "判定"]),
        "Start/End": get_master(stencil, ["Start/End", "开始/结束"]),
    }

    doc = app.Documents.Add("")
    page = doc.Pages.Item(1)

    page_width = float(scene["page"]["width"])
    page_height = float(scene["page"]["height"])
    try_set_formula(page.PageSheet, "PageWidth", f"{page_width} in")
    try_set_formula(page.PageSheet, "PageHeight", f"{page_height} in")

    nodes_by_id = {node["id"]: node for node in scene.get("nodes", [])}
    asset_paths = {
        asset["id"]: Path(asset["path"]).resolve()
        for asset in scene.get("assets", [])
        if asset.get("path")
    }

    for node in sorted(scene.get("nodes", []), key=lambda item: item.get("z", 0)):
        draw_node(page, page_height, node, asset_paths, component_map, profile, masters)

    for edge in sorted(scene.get("edges", []), key=lambda item: item.get("z", 100)):
        _, mid_x, mid_y = draw_edge(page, page_height, edge, nodes_by_id, component_map, profile)
        if edge.get("label"):
            draw_edge_label(page, page_height, str(edge["label"]), mid_x, mid_y, profile)

    basename = args.basename or scene_path.stem
    vsdx_path = output_dir / f"{basename}.vsdx"
    svg_path = output_dir / f"{basename}.svg"
    png_path = output_dir / f"{basename}.png"

    doc.SaveAs(str(vsdx_path))

    export_errors = []
    for export_path in (svg_path, png_path):
        try:
            page.Export(str(export_path))
        except Exception as exc:
            export_errors.append(f"{export_path.name}: {exc}")

    print(f"Style profile: {profile_name}")
    print(f"Wrote: {vsdx_path}")
    if export_errors:
        print("Export warnings:")
        for item in export_errors:
            print(f"- {item}")
    else:
        print(f"Wrote: {svg_path}")
        print(f"Wrote: {png_path}")

    if not args.keep_open:
        try:
            doc.Saved = True
            doc.Close()
        except Exception:
            pass
        if stencil is not None:
            try:
                stencil.Close()
            except Exception:
                pass
        app.Quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
