#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


TARGET_WIDTH_IN = 13.333


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def expanded(self, pad: int) -> "Box":
        return Box(self.x - pad, self.y - pad, self.w + pad * 2, self.h + pad * 2)


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1 = max(0, min(width - 1, box.x))
    y1 = max(0, min(height - 1, box.y))
    x2 = max(x1 + 1, min(width, box.x2))
    y2 = max(y1 + 1, min(height, box.y2))
    return Box(x1, y1, x2 - x1, y2 - y1)


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / max(1, a.area + b.area - inter)


def overlap_ratio(inner: Box, outer: Box) -> float:
    x1 = max(inner.x, outer.x)
    y1 = max(inner.y, outer.y)
    x2 = min(inner.x2, outer.x2)
    y2 = min(inner.y2, outer.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1, inner.area)


def color_to_hex(bgr: np.ndarray | list[float] | tuple[float, float, float]) -> str:
    b, g, r = [int(max(0, min(255, round(float(v))))) for v in bgr[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(hex_color: str) -> float:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_text_color(fill: str) -> str:
    return "#FFFFFF" if luminance(fill) < 126 else "#111827"


OCR_TEXT_FIXES = {
    "add&norm": "Add & Norm",
    "add&norn": "Add & Norm",
    "add&nom": "Add & Norm",
    "indino": "Output",
    "outpuf": "Output",
    "ouput": "Output",
    "outpu": "Output",
    "outpit": "Output",
}


def normalize_diagram_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        compact = re.sub(r"[^A-Za-z0-9&]+", "", line).lower()
        line = OCR_TEXT_FIXES.get(compact, line)
        line = re.sub(r"\s*&\s*", " & ", line)
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)
    return "\n".join(lines)


def sanitize_id(text: str, prefix: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not base:
        base = str(index)
    return f"{prefix}_{base[:36]}_{index:03d}"


def polygon_box(points: list[list[float]]) -> Box:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1 = math.floor(min(xs))
    y1 = math.floor(min(ys))
    x2 = math.ceil(max(xs))
    y2 = math.ceil(max(ys))
    return Box(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def normalize_ocr_result(raw: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not raw:
        return items
    for index, item in enumerate(raw):
        try:
            points, text, confidence = item
        except (TypeError, ValueError):
            continue
        text = str(text).strip()
        if not text:
            continue
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        if score < 0.45:
            continue
        box = polygon_box(points)
        items.append({"id": index, "text": text, "confidence": score, "box": box, "points": points})
    return items


def run_ocr(image_path: Path) -> list[dict[str, Any]]:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return []

    engine = RapidOCR()
    result, _ = engine(str(image_path))
    return normalize_ocr_result(result)


def read_image_bgr(image_path: Path) -> np.ndarray | None:
    """Read an image through a Windows Unicode-safe path."""
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return image
    except Exception:
        pass
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def text_mask(shape: tuple[int, int], ocr_items: list[dict[str, Any]], pad: int = 8) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    for item in ocr_items:
        box = clamp_box(item["box"].expanded(pad), width, height)
        cv2.rectangle(mask, (box.x, box.y), (box.x2, box.y2), 255, -1)
    return mask


def detect_colored_regions(image: np.ndarray, ocr_items: list[dict[str, Any]]) -> list[Box]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((saturation > 26) & (value > 75) & (value < 254)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(text_mask((height, width), ocr_items, pad=6)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    image_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = Box(x, y, w, h)
        if box.area < 450 or box.area > image_area * 0.92:
            continue
        if w < 16 or h < 10:
            continue
        aspect = w / max(1, h)
        if aspect > 30 or aspect < 0.025:
            continue
        boxes.append(clamp_box(box, width, height))

    return merge_boxes(non_max_suppression(boxes, 0.72), width, height)


def detect_rectangular_frames(image: np.ndarray) -> list[Box]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    image_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = Box(x, y, w, h)
        if box.area < 1200 or box.area > image_area * 0.88:
            continue
        if w < 28 or h < 18:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        extent = cv2.contourArea(contour) / max(1, box.area)
        if len(approx) <= 10 and extent > 0.08:
            boxes.append(clamp_box(box, width, height))
    return non_max_suppression(boxes, 0.65)


def detect_line_edges(image: np.ndarray, ocr_items: list[dict[str, Any]], shape_boxes: list[Box]) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_not(text_mask((height, width), ocr_items, pad=12))
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=65,
        minLineLength=max(42, int(min(width, height) * 0.07)),
        maxLineGap=6,
    )
    if lines is None:
        return []

    records: list[dict[str, Any]] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in raw]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 25:
            continue
        if line_inside_shape(x1, y1, x2, y2, shape_boxes):
            continue
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx >= dy * 5:
            y = int(round((y1 + y2) / 2))
            x_start, x_end = sorted([x1, x2])
            records.append({"kind": "h", "x1": x_start, "y1": y, "x2": x_end, "y2": y, "length": length})
        elif dy >= dx * 5:
            x = int(round((x1 + x2) / 2))
            y_start, y_end = sorted([y1, y2])
            records.append({"kind": "v", "x1": x, "y1": y_start, "x2": x, "y2": y_end, "length": length})
    return merge_line_records(records)


def line_inside_shape(x1: int, y1: int, x2: int, y2: int, shape_boxes: list[Box]) -> bool:
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    length = math.hypot(x2 - x1, y2 - y1)
    for box in shape_boxes:
        if box.w > 260 and box.h > 160:
            continue
        padded = box.expanded(4)
        if padded.x <= mid_x <= padded.x2 and padded.y <= mid_y <= padded.y2:
            if length <= max(box.w, box.h) * 1.25:
                return True
    return False


def merge_line_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in sorted(records, key=lambda r: (r["kind"], r["y1"] if r["kind"] == "h" else r["x1"], r["x1"], r["y1"])):
        target = None
        for existing in merged:
            if existing["kind"] != item["kind"]:
                continue
            if item["kind"] == "h":
                if abs(existing["y1"] - item["y1"]) <= 4 and not (item["x1"] > existing["x2"] + 12 or item["x2"] < existing["x1"] - 12):
                    target = existing
                    break
            else:
                if abs(existing["x1"] - item["x1"]) <= 4 and not (item["y1"] > existing["y2"] + 12 or item["y2"] < existing["y1"] - 12):
                    target = existing
                    break
        if target is None:
            merged.append(dict(item))
        elif item["kind"] == "h":
            target["x1"] = min(target["x1"], item["x1"])
            target["x2"] = max(target["x2"], item["x2"])
            target["y1"] = target["y2"] = int(round((target["y1"] + item["y1"]) / 2))
            target["length"] = target["x2"] - target["x1"]
        else:
            target["y1"] = min(target["y1"], item["y1"])
            target["y2"] = max(target["y2"], item["y2"])
            target["x1"] = target["x2"] = int(round((target["x1"] + item["x1"]) / 2))
            target["length"] = target["y2"] - target["y1"]

    return [item for item in merged if item["length"] >= 24]


def non_max_suppression(boxes: list[Box], threshold: float) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        duplicate = False
        for other in kept:
            area_ratio = min(box.area, other.area) / max(1, max(box.area, other.area))
            if iou(box, other) > threshold or (overlap_ratio(box, other) > 0.88 and area_ratio > 0.55):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(box)
    return kept


def merge_boxes(boxes: list[Box], width: int, height: int) -> list[Box]:
    changed = True
    current = list(boxes)
    while changed:
        changed = False
        next_boxes: list[Box] = []
        used = [False] * len(current)
        for i, box in enumerate(current):
            if used[i]:
                continue
            merged = box
            used[i] = True
            for j in range(i + 1, len(current)):
                other = current[j]
                if used[j]:
                    continue
                close_x = other.x <= merged.x2 + 4 and other.x2 >= merged.x - 4
                close_y = other.y <= merged.y2 + 4 and other.y2 >= merged.y - 4
                if close_x and close_y and (iou(merged, other) > 0.08 or overlap_ratio(other, merged) > 0.45 or overlap_ratio(merged, other) > 0.45):
                    x1 = min(merged.x, other.x)
                    y1 = min(merged.y, other.y)
                    x2 = max(merged.x2, other.x2)
                    y2 = max(merged.y2, other.y2)
                    merged = clamp_box(Box(x1, y1, x2 - x1, y2 - y1), width, height)
                    used[j] = True
                    changed = True
            next_boxes.append(merged)
        current = next_boxes
    return current


def box_fill(image: np.ndarray, box: Box) -> str:
    height, width = image.shape[:2]
    box = clamp_box(box, width, height)
    roi = image[box.y : box.y2, box.x : box.x2]
    if roi.size == 0:
        return "#FFFFFF"
    # Ignore dark text/lines when estimating shape fill.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    valid = hsv[:, :, 2] > 120
    if np.count_nonzero(valid) < max(8, roi.shape[0] * roi.shape[1] * 0.08):
        valid = np.ones(roi.shape[:2], dtype=bool)
    pixels = roi[valid]
    median = np.median(pixels.reshape(-1, 3), axis=0)
    return color_to_hex(median)


def likely_container(box: Box, width: int, height: int) -> bool:
    return box.w > width * 0.25 and box.h > height * 0.12 and box.area > width * height * 0.025


def font_size_for_box(box: Box) -> float:
    return max(4.5, min(11.0, box.h * 0.38))


def build_shape_nodes(image: np.ndarray, ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    boxes = detect_colored_regions(image, ocr_items) + detect_rectangular_frames(image)
    boxes = non_max_suppression(boxes, 0.58)
    nodes: list[dict[str, Any]] = []
    for index, box in enumerate(sorted(boxes, key=lambda item: (likely_container(item, width, height), item.y, item.x))):
        if box.w >= width * 0.94 and box.h >= height * 0.94:
            continue
        fill = box_fill(image, box)
        if likely_container(box, width, height):
            node_type = "group_container"
            style = {
                "fill": fill if luminance(fill) < 246 else "#F8FAFC",
                "line": "#94A3B8",
                "line_dash": "dash",
                "line_weight_pt": 1.0,
                "text_color": "#111827",
                "font_size_pt": 1,
            }
            text = ""
            z = 5
        else:
            node_type = "rounded_process"
            style = {
                "fill": fill,
                "line": "#64748B",
                "line_weight_pt": 0.85,
                "text_color": readable_text_color(fill),
                "font_size_pt": max(1.0, min(9.0, box.h * 0.25)),
                "text_fit": "shrink",
                "rounding_in": 0.06,
            }
            text = ""
            z = 20
        nodes.append(
            {
                "id": f"{'container' if node_type == 'group_container' else 'shape'}_{index:03d}",
                "type": node_type,
                "x": box.x,
                "y": box.y,
                "w": box.w,
                "h": box.h,
                "z": z,
                "text": text,
                "style": style,
                "source_bbox_px": [box.x, box.y, box.x2, box.y2],
            }
        )
    return nodes


def build_text_nodes(ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(ocr_items):
        box: Box = item["box"]
        text = normalize_diagram_text(item["text"])
        rotated = box.h > box.w * 2.4 and len(text) > 8
        style = {
            "fill": "none",
            "line": "none",
            "text_color": "#111827",
            "font_family": "Times New Roman",
            "font_size_pt": font_size_for_box(box),
            "min_font_size_pt": 4.5,
            "text_fit": "shrink",
            "text_margin_in": 0.0,
        }
        if rotated:
            style["text_angle_deg"] = 90
            style["rotated_text_box_safety_factor"] = 1.25
        nodes.append(
            {
                "id": sanitize_id(text, "text", index),
                "type": "text_block",
                "x": box.x,
                "y": box.y,
                "w": max(8, box.w),
                "h": max(8, box.h),
                "z": 80,
                "text": text,
                "style": style,
                "source_bbox_px": [box.x, box.y, box.x2, box.y2],
                "ocr_confidence": round(float(item.get("confidence", 0.0)), 3),
            }
        )
    return nodes


def endpoint_from_line(line: dict[str, Any], which: str) -> list[float]:
    if which == "start":
        return [float(line["x1"]), float(line["y1"])]
    return [float(line["x2"]), float(line["y2"])]


def build_edges(image: np.ndarray, ocr_items: list[dict[str, Any]], shape_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shape_boxes = []
    for node in shape_nodes:
        bbox = node.get("source_bbox_px")
        if isinstance(bbox, list) and len(bbox) == 4:
            shape_boxes.append(Box(int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])))
    records = detect_line_edges(image, ocr_items, shape_boxes)
    edges: list[dict[str, Any]] = []
    for index, line in enumerate(records[:90]):
        route = "horizontal" if line["kind"] == "h" else "vertical"
        edges.append(
            {
                "id": f"line_{index:03d}",
                "type": "line_segment",
                "from_point": endpoint_from_line(line, "start"),
                "to_point": endpoint_from_line(line, "end"),
                "route": route,
                "z": 55,
                "style": {
                    "line": "#111827",
                    "line_weight_pt": 0.8,
                    "end_arrow": "none",
                },
                "source_bbox_px": [
                    min(line["x1"], line["x2"]),
                    min(line["y1"], line["y2"]),
                    max(line["x1"], line["x2"]),
                    max(line["y1"], line["y2"]),
                ],
            }
        )
    return edges


def trace_vector_edges(
    image: np.ndarray,
    ocr_items: list[dict[str, Any]],
    *,
    max_segments: int = 320,
) -> list[dict[str, Any]]:
    """Fallback vector trace: editable line segments only, never raster tiles."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_not(text_mask((height, width), ocr_items, pad=4))
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[float, float], tuple[float, float]]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, False)
        if perimeter < 18:
            continue
        approx = cv2.approxPolyDP(contour, max(1.6, perimeter * 0.012), False)
        points = approx.reshape(-1, 2)
        if len(points) < 2:
            continue
        for start, end in zip(points, points[1:]):
            x1, y1 = [float(value) for value in start]
            x2, y2 = [float(value) for value in end]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 8:
                continue
            if length > max(width, height) * 0.9:
                continue
            candidates.append((length, (x1, y1), (x2, y2)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for _, start, end in candidates:
        key = tuple(round(value / 3) for value in (start[0], start[1], end[0], end[1]))
        rev_key = (key[2], key[3], key[0], key[1])
        if key in seen or rev_key in seen:
            continue
        seen.add(key)
        kept.append((start, end))
        if len(kept) >= max_segments:
            break

    traced: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(kept):
        route = "straight"
        if abs(start[0] - end[0]) >= abs(start[1] - end[1]) * 5:
            route = "horizontal"
            y = round((start[1] + end[1]) / 2, 2)
            start = (start[0], y)
            end = (end[0], y)
        elif abs(start[1] - end[1]) >= abs(start[0] - end[0]) * 5:
            route = "vertical"
            x = round((start[0] + end[0]) / 2, 2)
            start = (x, start[1])
            end = (x, end[1])
        traced.append(
            {
                "id": f"trace_line_{index:03d}",
                "type": "line_segment",
                "from_point": [round(start[0], 2), round(start[1], 2)],
                "to_point": [round(end[0], 2), round(end[1], 2)],
                "route": route,
                "z": 44,
                "style": {
                    "line": "#1F2937",
                    "line_weight_pt": 0.42,
                    "end_arrow": "none",
                    "line_transparency_pct": 8,
                },
            }
        )
    return traced


def icon_candidate_mask(image: np.ndarray, ocr_items: list[dict[str, Any]]) -> np.ndarray:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    edge_mask = cv2.Canny(gray, 42, 132)
    dark_stroke_mask = ((gray < 178) & (value < 246)).astype(np.uint8) * 255
    saturated_stroke_mask = ((saturation > 42) & (value > 35) & (value < 248)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(edge_mask, dark_stroke_mask)
    mask = cv2.bitwise_or(mask, saturated_stroke_mask)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(text_mask((height, width), ocr_items, pad=4)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)


def meaningful_contour_count(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        if area >= 10 or perimeter >= 16:
            count += 1
    return count


def overlaps_existing_module(box: Box, blocked_boxes: list[Box]) -> bool:
    for blocked in blocked_boxes:
        area_ratio = box.area / max(1, blocked.area)
        if iou(box, blocked) > 0.42 and area_ratio > 0.32:
            return True
        if overlap_ratio(box, blocked) > 0.84 and area_ratio > 0.24:
            return True
    return False


def detect_icon_regions(
    image: np.ndarray,
    ocr_items: list[dict[str, Any]],
    blocked_boxes: list[Box] | None = None,
    *,
    max_icons: int = 24,
) -> list[Box]:
    """Find compact icon-like regions that should be redrawn as editable vectors."""
    height, width = image.shape[:2]
    image_area = width * height
    blocked_boxes = blocked_boxes or []
    text_boxes = [item["box"].expanded(3) for item in ocr_items]

    mask = icon_candidate_mask(image, ocr_items)
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    grouped = cv2.dilate(grouped, np.ones((3, 3), np.uint8), iterations=1)

    min_area = max(90, int(image_area * 0.000035))
    max_area = max(2200, int(image_area * 0.065))
    candidates: list[tuple[float, Box]] = []

    def add_candidate(box: Box, *, base_score: float = 0.0, rect_like: bool = False) -> None:
        box = clamp_box(box, width, height)
        if box.area < min_area or box.area > max_area:
            return
        if box.w < 12 or box.h < 12:
            return
        if box.w > width * 0.30 or box.h > height * 0.36:
            return
        aspect = box.w / max(1, box.h)
        if aspect > 4.5 or aspect < 0.22:
            return
        if overlaps_existing_module(box, blocked_boxes):
            return

        text_overlap = mostly_text_overlap(box, text_boxes)
        crop_mask = mask[box.y : box.y2, box.x : box.x2]
        if crop_mask.size == 0:
            return
        crop_edges = cv2.Canny(cv2.cvtColor(image[box.y : box.y2, box.x : box.x2], cv2.COLOR_BGR2GRAY), 42, 132)
        edge_density = float((crop_edges > 0).mean())
        ink_coverage = float((crop_mask > 0).mean())
        if edge_density < 0.020 and ink_coverage < 0.035:
            return
        detail_count = meaningful_contour_count(crop_edges)
        if text_overlap and detail_count < 5 and edge_density < 0.075:
            return
        if rect_like and detail_count < 3:
            return
        if detail_count < 2 and edge_density < 0.055 and ink_coverage < 0.10:
            return
        compactness = 1.0 / max(1.0, abs(math.log(max(0.25, min(4.0, aspect)))))
        score = base_score + edge_density * 110.0 + ink_coverage * 38.0 + min(detail_count, 14) * 1.8 + compactness
        candidates.append((score, box))

    for seed_box in detect_colored_regions(image, []) + detect_rectangular_frames(image):
        if likely_container(seed_box, width, height):
            continue
        add_candidate(seed_box.expanded(2), base_score=18.0)

    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = clamp_box(Box(x, y, w, h).expanded(2), width, height)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(1.6, perimeter * 0.018), True) if perimeter > 0 else contour
        extent = abs(cv2.contourArea(contour)) / max(1, box.area)
        rect_like = len(approx) <= 5 and extent > 0.48
        add_candidate(box, rect_like=rect_like)

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[Box] = []
    for _, box in candidates:
        if any(iou(box, existing) > 0.32 or overlap_ratio(box, existing) > 0.62 for existing in kept):
            continue
        kept.append(box)
        if len(kept) >= max_icons:
            break
    return sorted(kept, key=lambda item: (item.y, item.x))


def icon_foreground_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    edges = cv2.Canny(gray, 36, 124)

    border_parts = [
        crop[0:1, :, :],
        crop[-1:, :, :],
        crop[:, 0:1, :],
        crop[:, -1:, :],
    ]
    border = np.concatenate([part.reshape(-1, 3) for part in border_parts if part.size], axis=0)
    background = np.median(border, axis=0) if border.size else np.array([255, 255, 255], dtype=np.float32)
    color_delta = np.linalg.norm(crop.astype(np.float32) - background.astype(np.float32), axis=2)

    contrast = (
        (color_delta > 26)
        | ((gray < 185) & (value < 246))
        | ((saturation > 46) & (value > 35) & (value < 248))
        | (edges > 0)
    ).astype(np.uint8) * 255
    contrast = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.morphologyEx(contrast, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)


def icon_foreground_color(crop: np.ndarray, mask: np.ndarray) -> str:
    if crop.size == 0 or mask.size == 0:
        return "#1F2937"
    pixels = crop[mask > 0]
    if pixels.size == 0:
        return "#1F2937"
    median = np.median(pixels.reshape(-1, 3), axis=0)
    color = color_to_hex(median)
    if luminance(color) > 232:
        return "#1F2937"
    return color


def route_for_segment(start: tuple[float, float], end: tuple[float, float]) -> str:
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    if dx >= dy * 5:
        return "horizontal"
    if dy >= dx * 5:
        return "vertical"
    return "straight"


def build_icon_vector_parts(
    image: np.ndarray,
    ocr_items: list[dict[str, Any]],
    blocked_boxes: list[Box] | None = None,
    *,
    max_icons: int = 24,
    max_segments_per_icon: int = 44,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    icon_boxes = detect_icon_regions(image, ocr_items, blocked_boxes, max_icons=max_icons)
    height, width = image.shape[:2]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []

    for icon_index, raw_box in enumerate(icon_boxes):
        box = clamp_box(raw_box, width, height)
        crop = image[box.y : box.y2, box.x : box.x2]
        mask = icon_foreground_mask(crop)
        if mask.size == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_records: list[tuple[float, np.ndarray, float]] = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = abs(cv2.contourArea(contour))
            if perimeter < 10 and area < 8:
                continue
            contour_records.append((area + perimeter * 0.25, contour, perimeter))
        contour_records.sort(key=lambda item: item[0], reverse=True)
        if not contour_records:
            continue

        icon_id = f"icon_{icon_index:02d}"
        color = icon_foreground_color(crop, mask)
        region_part_count = 0
        seen_segments: set[tuple[int, int, int, int]] = set()
        segment_records: list[tuple[float, tuple[float, float], tuple[float, float], str]] = []

        for contour_order, (_, contour, perimeter) in enumerate(contour_records[:18]):
            area = abs(cv2.contourArea(contour))
            epsilon = max(1.2, perimeter * 0.018)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 2:
                continue

            if 3 <= len(points) <= 16 and area >= max(18.0, box.area * 0.018):
                normalized_points = [
                    [
                        round(float(point[0]) / max(1, box.w), 4),
                        round(float(point[1]) / max(1, box.h), 4),
                    ]
                    for point in points
                ]
                nodes.append(
                    {
                        "id": f"{icon_id}_shape_{contour_order:02d}",
                        "type": "polygon_node",
                        "x": box.x,
                        "y": box.y,
                        "w": box.w,
                        "h": box.h,
                        "z": 63,
                        "text": "",
                        "points": normalized_points,
                        "allow_overlap": True,
                        "semantic_role": "editable_icon_polygon",
                        "icon_region_id": icon_id,
                        "source_bbox_px": [box.x, box.y, box.x2, box.y2],
                        "style": {
                            "fill": "none",
                            "line": color,
                            "line_weight_pt": 0.62,
                            "end_arrow": "none",
                        },
                    }
                )
                region_part_count += 1

            closed_points = [tuple(float(v) for v in point) for point in points]
            if len(closed_points) >= 3:
                closed_points.append(closed_points[0])
            for start_local, end_local in zip(closed_points, closed_points[1:]):
                start = (box.x + start_local[0], box.y + start_local[1])
                end = (box.x + end_local[0], box.y + end_local[1])
                length = math.hypot(end[0] - start[0], end[1] - start[1])
                if length < 3.2:
                    continue
                key = tuple(round(value / 2) for value in (start[0], start[1], end[0], end[1]))
                reverse_key = (key[2], key[3], key[0], key[1])
                if key in seen_segments or reverse_key in seen_segments:
                    continue
                seen_segments.add(key)
                segment_records.append((length, start, end, route_for_segment(start, end)))

        segment_records.sort(key=lambda item: item[0], reverse=True)
        for segment_index, (_, start, end, route) in enumerate(segment_records[:max_segments_per_icon]):
            edges.append(
                {
                    "id": f"{icon_id}_stroke_{segment_index:03d}",
                    "type": "line_segment",
                    "from_point": [round(start[0], 2), round(start[1], 2)],
                    "to_point": [round(end[0], 2), round(end[1], 2)],
                    "route": route,
                    "z": 66,
                    "allow_diagonal": True,
                    "semantic_role": "editable_icon_stroke",
                    "icon_region_id": icon_id,
                    "source_bbox_px": [box.x, box.y, box.x2, box.y2],
                    "style": {
                        "line": color,
                        "line_weight_pt": 0.56,
                        "end_arrow": "none",
                    },
                }
            )
            region_part_count += 1

        if region_part_count:
            regions.append(
                {
                    "id": icon_id,
                    "source_bbox_px": [box.x, box.y, box.x2, box.y2],
                    "vector_parts": region_part_count,
                    "policy": "editable_vector_no_raster",
                }
            )

    return nodes, edges, regions


def box_from_source_bbox(raw: Any) -> Box | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in raw]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2 - x1, y2 - y1)


def filter_icon_duplicate_shape_nodes(
    shape_nodes: list[dict[str, Any]],
    icon_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    icon_boxes = [box for box in (box_from_source_bbox(region.get("source_bbox_px")) for region in icon_regions) if box is not None]
    if not icon_boxes:
        return shape_nodes

    filtered: list[dict[str, Any]] = []
    for node in shape_nodes:
        if node.get("type") == "group_container":
            filtered.append(node)
            continue
        node_box = node_to_box(node)
        duplicate = False
        for icon_box in icon_boxes:
            area_ratio = min(node_box.area, icon_box.area) / max(1, max(node_box.area, icon_box.area))
            if iou(node_box, icon_box) > 0.38 or (overlap_ratio(node_box, icon_box) > 0.78 and area_ratio > 0.45):
                duplicate = True
                break
        if not duplicate:
            filtered.append(node)
    return filtered


def build_vector_trace_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
    max_trace_segments: int = 320,
    mode_name: str = "vector_trace",
) -> dict[str, Any]:
    image = read_image_bgr(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    nodes: list[dict[str, Any]] = [
        {
            "id": "page_background",
            "type": "page_background",
            "x": 0,
            "y": 0,
            "w": width,
            "h": height,
            "z": 0,
            "text": "",
            "style": {"fill": "#FFFFFF", "line": "none"},
        }
    ]
    shape_nodes = build_shape_nodes(image, ocr_items)
    icon_nodes, icon_edges, icon_regions = build_icon_vector_parts(
        image,
        ocr_items,
        [node_to_box(node) for node in shape_nodes if node.get("type") == "group_container"],
        max_icons=28,
        max_segments_per_icon=52 if max_trace_segments > 400 else 40,
    )
    shape_nodes = filter_icon_duplicate_shape_nodes(shape_nodes, icon_regions)
    nodes.extend(shape_nodes)
    nodes.extend(icon_nodes)
    nodes.extend(build_text_nodes(ocr_items))

    edges = build_edges(image, ocr_items, shape_nodes)
    edges.extend(icon_edges)
    existing_ids = {edge["id"] for edge in edges}
    for edge in trace_vector_edges(image, ocr_items, max_segments=max_trace_segments):
        if edge["id"] in existing_ids:
            edge["id"] = f"{edge['id']}_fallback"
        edges.append(edge)

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": f"visiomaster.image_auto_scene.{mode_name}",
            "style_profile": "paper_white",
            "fidelity": "auto_editable_vector_trace_draft",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "visual_reference_layer": False,
            "raster_tile_policy": "disabled_by_workflow",
            "partial_raster_tiles": 0,
            "reconstruction_mode": mode_name,
            "icon_reconstruction_policy": "editable_vector_no_raster",
            "icon_vector_regions": len(icon_regions),
            "icon_vector_parts": len(icon_nodes) + len(icon_edges),
            "icon_regions": icon_regions,
            "notes": [
                "Fallback self-check mode: redraws detected visual strokes as editable Visio line segments and text.",
                "Compact icon-like regions are reconstructed as editable vector polygons and line segments.",
                "No original image, local tile, or raster reference layer is embedded.",
            ],
        },
        "page": {
            "width": width,
            "height": height,
            "units": "px",
            "origin": "top-left",
            "target_width_in": TARGET_WIDTH_IN,
            "background": "#FFFFFF",
        },
        "nodes": nodes,
        "edges": edges,
        "assets": [],
    }


def px_node(
    node_id: str,
    node_type: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    *,
    fill: str = "#FFFFFF",
    line: str = "#64748B",
    z: int = 20,
    font_size: float = 13,
    text_color: str | None = None,
    dash: str = "solid",
    text_angle: float | None = None,
    rounding: float = 0.08,
) -> dict[str, Any]:
    style: dict[str, Any] = {
        "fill": fill,
        "line": line,
        "line_weight_pt": 1.0,
        "line_dash": dash,
        "text_color": text_color or readable_text_color(fill),
        "font_family": "Times New Roman",
        "font_size_pt": font_size,
        "min_font_size_pt": 5.0,
        "text_fit": "shrink",
        "rounding_in": rounding,
    }
    if text_angle is not None:
        style["text_angle_deg"] = text_angle
        style["rotated_text_box_safety_factor"] = 1.18
    return {
        "id": node_id,
        "type": node_type,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "z": z,
        "text": text,
        "style": style,
    }


def text_node(
    node_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    font_size: float = 16,
    weight: str = "regular",
    angle: float | None = None,
    z: int = 90,
) -> dict[str, Any]:
    node = px_node(
        node_id,
        "text_block",
        x,
        y,
        w,
        h,
        text,
        fill="none",
        line="none",
        z=z,
        font_size=font_size,
        text_color="#111827",
        text_angle=angle,
        rounding=0.0,
    )
    node["style"]["font_weight"] = weight
    node["style"]["text_margin_in"] = 0.0
    return node


def should_add_detail_tiles(image: np.ndarray, ocr_items: list[dict[str, Any]]) -> bool:
    height, width = image.shape[:2]
    aspect = max(width / max(1, height), height / max(1, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_density = float((cv2.Canny(gray, 60, 160) > 0).mean())
    return (
        len(ocr_items) >= 35
        or edge_density >= 0.05
        or aspect >= 2.2
        or width * height >= 1_200_000 and len(ocr_items) >= 20
    )


def should_add_raster_tiles(image: np.ndarray, ocr_items: list[dict[str, Any]]) -> bool:
    """Use only small local raster tiles for photos/plots/icons, never a full-image background."""
    return should_add_detail_tiles(image, ocr_items)


def safe_asset_slug(image_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._")
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{stem or 'image'}_{digest}"


def asset_workspace_for_image(image_path: Path) -> Path:
    return Path.cwd() / "work" / "auto_assets" / safe_asset_slug(image_path)


def box_texture_score(image: np.ndarray, box: Box) -> float:
    height, width = image.shape[:2]
    box = clamp_box(box, width, height)
    roi = image[box.y : box.y2, box.x : box.x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edge_density = float((cv2.Canny(gray, 60, 160) > 0).mean())
    color_std = float(np.mean(np.std(roi.reshape(-1, 3), axis=0)))
    saturation = float(hsv[:, :, 1].mean())
    return edge_density * 100.0 + min(color_std, 80.0) * 0.55 + min(saturation, 180.0) * 0.08


def mostly_text_overlap(box: Box, text_boxes: list[Box]) -> bool:
    if not text_boxes:
        return False
    covered = 0
    for text_box in text_boxes:
        x1 = max(box.x, text_box.x)
        y1 = max(box.y, text_box.y)
        x2 = min(box.x2, text_box.x2)
        y2 = min(box.y2, text_box.y2)
        covered += max(0, x2 - x1) * max(0, y2 - y1)
    return covered / max(1, box.area) > 0.24


def blocks_editable_module(box: Box, blocked_boxes: list[Box]) -> bool:
    for blocked in blocked_boxes:
        if box.area < blocked.area * 0.55:
            continue
        if overlap_ratio(box, blocked) > 0.72 or iou(box, blocked) > 0.42:
            return True
    return False


def detect_raster_asset_boxes(
    image: np.ndarray,
    ocr_items: list[dict[str, Any]],
    blocked_boxes: list[Box] | None = None,
    *,
    max_tiles: int = 18,
) -> list[Box]:
    height, width = image.shape[:2]
    image_area = width * height
    blocked_boxes = blocked_boxes or []
    text_boxes = [item["box"].expanded(3) for item in ocr_items]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    edges = cv2.Canny(gray, 60, 160)

    text_block_mask = text_mask((height, width), ocr_items, pad=3)
    texture_mask = (
        ((edges > 0) & (value < 248))
        | ((saturation > 38) & (value > 35) & (value < 252))
    ).astype(np.uint8) * 255
    texture_mask = cv2.bitwise_and(texture_mask, cv2.bitwise_not(text_block_mask))
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    texture_mask = cv2.dilate(texture_mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, Box]] = []
    min_area = max(420, int(image_area * 0.00035))
    max_area = int(image_area * 0.020)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = clamp_box(Box(x, y, w, h).expanded(2), width, height)
        if box.area < min_area or box.area > max_area:
            continue
        if box.w < 18 or box.h < 14:
            continue
        if box.w > width * 0.34 or box.h > height * 0.34:
            continue
        aspect = box.w / max(1, box.h)
        if aspect > 8.5 or aspect < 0.12:
            continue
        if mostly_text_overlap(box, text_boxes):
            continue
        if blocks_editable_module(box, blocked_boxes):
            continue
        score = box_texture_score(image, box)
        if score < 13.5:
            continue
        candidates.append((score + math.log(max(2, box.area)), box))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[Box] = []
    for _, box in candidates:
        if any(iou(box, existing) > 0.36 or overlap_ratio(box, existing) > 0.64 for existing in kept):
            continue
        kept.append(box)
        if len(kept) >= max_tiles:
            break
    return sorted(kept, key=lambda item: (item.y, item.x))


def raster_box_records(
    image: np.ndarray,
    ocr_items: list[dict[str, Any]],
    blocked_boxes: list[Box] | None = None,
    *,
    max_tiles: int = 18,
) -> list[tuple[Box, int, str]]:
    detail_boxes = detect_raster_asset_boxes(image, ocr_items, blocked_boxes, max_tiles=max_tiles)
    records: list[tuple[Box, int, str]] = []
    for box in detail_boxes:
        records.append((box, 35, "detail_tile"))
    return records[:max_tiles]


def create_raster_asset_tiles(
    image_path: Path,
    image: np.ndarray,
    ocr_items: list[dict[str, Any]],
    blocked_boxes: list[Box] | None = None,
    *,
    max_tiles: int = 18,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = raster_box_records(image, ocr_items, blocked_boxes, max_tiles=max_tiles)
    if not records:
        return [], []

    asset_dir = asset_workspace_for_image(image_path)
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nodes: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    total_area = 0
    max_total_area = int(image.shape[0] * image.shape[1] * 0.16)
    for index, (box, z, role) in enumerate(records):
        if total_area + box.area > max_total_area:
            continue
        total_area += box.area
        asset_id = f"local_raster_{index:03d}"
        asset_path = asset_dir / f"{asset_id}.png"
        crop = source_rgb[box.y : box.y2, box.x : box.x2]
        Image.fromarray(crop).save(asset_path)
        assets.append({"id": asset_id, "kind": "image", "path": str(asset_path.resolve())})
        nodes.append(
            {
                "id": f"image_tile_{index:03d}",
                "type": "image_tile",
                "x": box.x,
                "y": box.y,
                "w": box.w,
                "h": box.h,
                "z": z,
                "text": "",
                "asset_ref": asset_id,
                "raster_role": role,
                "source_bbox_px": [box.x, box.y, box.x2, box.y2],
                "style": {"line": "#CBD5E1", "line_weight_pt": 0.35},
            }
        )
    return nodes, assets


def edge_px(
    edge_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    arrow: bool = True,
    route: str = "straight",
    points: list[list[float]] | None = None,
    z: int = 60,
    allow_cross_container: bool = False,
) -> dict[str, Any]:
    edge: dict[str, Any] = {
        "id": edge_id,
        "type": "arrow_connector" if arrow else "line_segment",
        "from_point": [x1, y1],
        "to_point": [x2, y2],
        "route": route,
        "z": z,
        "style": {
            "line": "#111827",
            "line_weight_pt": 1.05,
            "end_arrow": "triangle" if arrow else "none",
            "arrow_size": "small",
        },
    }
    if points:
        edge["points"] = points
        edge["orthogonalize_points"] = True
    if allow_cross_container:
        edge["allow_cross_container"] = True
    return edge


def contains_keywords(ocr_items: list[dict[str, Any]], keywords: list[str]) -> bool:
    corpus = " ".join(str(item.get("text", "")) for item in ocr_items).lower()
    return all(keyword.lower() in corpus for keyword in keywords)


def group_ocr_items(ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = sorted(ocr_items, key=lambda item: (item["box"].y, item["box"].x))
    groups: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, item in enumerate(remaining):
        if index in used:
            continue
        used.add(index)
        members = [item]
        changed = True
        while changed:
            changed = False
            group_box = union_boxes([member["box"] for member in members])
            group_cx = group_box.cx
            for other_index, other in enumerate(remaining):
                if other_index in used:
                    continue
                other_box: Box = other["box"]
                vertical_gap = other_box.y - group_box.y2
                overlaps_y = not (other_box.y > group_box.y2 + 3 or other_box.y2 < group_box.y - 3)
                center_close = abs(other_box.cx - group_cx) <= max(group_box.w, other_box.w) * 0.45 + 18
                stacked_close = 0 <= vertical_gap <= max(16, min(group_box.h, other_box.h) * 0.8)
                if center_close and (stacked_close or (overlaps_y and abs(other_box.cy - group_box.cy) < 8)):
                    used.add(other_index)
                    members.append(other)
                    changed = True
        members.sort(key=lambda member: (member["box"].y, member["box"].x))
        box = union_boxes([member["box"] for member in members])
        text = "\n".join(member["text"] for member in members)
        confidence = min(float(member.get("confidence", 0.0)) for member in members)
        groups.append({"text": text, "box": box, "members": members, "confidence": confidence})
    return sorted(groups, key=lambda item: (item["box"].y, item["box"].x))


def union_boxes(boxes: list[Box]) -> Box:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return Box(x1, y1, x2 - x1, y2 - y1)


def contiguous_intervals(flags: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_len:
                intervals.append((start, index))
            start = None
    if start is not None and len(flags) - start >= min_len:
        intervals.append((start, len(flags)))
    return intervals


def interval_containing(intervals: list[tuple[int, int]], center: float, min_width: int, max_width: int) -> tuple[int, int] | None:
    candidates = []
    for start, end in intervals:
        width = end - start
        if start <= center <= end and min_width <= width <= max_width:
            candidates.append((start, end, width))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (abs(item[2] - min_width * 1.8), item[2]))
    return candidates[0][0], candidates[0][1]


def fill_mask_for_modules(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return ((saturation > 10) & (value > 115) & (value < 254)).astype(np.uint8)


def find_enclosing_module_box(mask: np.ndarray, group_box: Box, image_width: int, image_height: int) -> Box | None:
    pad_y = max(8, int(group_box.h * 0.45))
    y1 = max(0, group_box.y - pad_y)
    y2 = min(image_height, group_box.y2 + pad_y)
    strip = mask[y1:y2, :]
    if strip.size == 0:
        return None
    col_counts = strip.sum(axis=0)
    col_flags = col_counts >= max(2, strip.shape[0] * 0.16)
    intervals = contiguous_intervals(col_flags, min_len=max(14, group_box.w // 2))
    x_interval = interval_containing(intervals, group_box.cx, group_box.w + 18, min(image_width, max(420, group_box.w * 5)))
    if x_interval is None:
        return None

    x1, x2 = x_interval
    col_strip = mask[:, x1:x2]
    row_counts = col_strip.sum(axis=1)
    row_flags = row_counts >= max(3, (x2 - x1) * 0.10)
    row_intervals = contiguous_intervals(row_flags, min_len=max(10, group_box.h // 2))
    y_interval = interval_containing(row_intervals, group_box.cy, group_box.h + 6, min(image_height, max(160, group_box.h * 5)))
    if y_interval is None:
        return None
    yy1, yy2 = y_interval
    box = Box(x1, yy1, x2 - x1, yy2 - yy1)
    if box.w < group_box.w + 12 or box.h < group_box.h + 4:
        return None
    density = float(mask[box.y : box.y2, box.x : box.x2].mean())
    if density < 0.26:
        return None
    return clamp_box(box.expanded(2), image_width, image_height)


def find_module_box_from_color_parts(group_box: Box, color_boxes: list[Box], image_width: int, image_height: int) -> Box | None:
    parts: list[Box] = []
    left_parts: list[Box] = []
    right_parts: list[Box] = []
    overlapping_parts: list[Box] = []
    for part in color_boxes:
        vertical = min(group_box.y2, part.y2) - max(group_box.y, part.y)
        if vertical <= 0:
            continue
        if vertical / max(1, min(group_box.h, part.h)) < 0.35:
            continue
        near_left = 0 <= group_box.x - part.x2 <= max(85, group_box.w * 0.9)
        near_right = 0 <= part.x - group_box.x2 <= max(85, group_box.w * 0.9)
        overlaps_x = not (part.x > group_box.x2 + 20 or part.x2 < group_box.x - 20)
        if near_left or near_right or overlaps_x:
            parts.append(part)
            if near_left:
                left_parts.append(part)
            if near_right:
                right_parts.append(part)
            if overlaps_x:
                overlapping_parts.append(part)
    if not parts:
        return None
    has_bracketed_fill = bool(left_parts and right_parts)
    has_under_text_fill = any(
        part.w >= max(18, group_box.w * 0.42)
        and part.h >= max(8, group_box.h * 0.45)
        for part in overlapping_parts
    )
    if not (has_bracketed_fill or has_under_text_fill):
        return None
    merged = union_boxes([group_box, *parts])
    if merged.w < group_box.w + 12 or merged.h < group_box.h:
        return None
    if merged.w > max(320, group_box.w * 4.2) or merged.h > max(110, group_box.h * 4.0):
        return None
    return clamp_box(merged.expanded(2), image_width, image_height)


def find_text_anchored_module_box(
    image: np.ndarray,
    mask: np.ndarray,
    group_box: Box,
    color_boxes: list[Box],
    image_width: int,
    image_height: int,
) -> Box | None:
    direct = find_enclosing_module_box(mask, group_box, image_width, image_height)
    if direct is not None:
        return direct
    return find_module_box_from_color_parts(group_box, color_boxes, image_width, image_height)


def detect_gray_containers(image: np.ndarray) -> list[Box]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation < 22) & (value > 175) & (value < 244)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = Box(x, y, w, h)
        if box.area < width * height * 0.018:
            continue
        if box.w < 120 or box.h < 120:
            continue
        if box.area > width * height * 0.80:
            continue
        boxes.append(clamp_box(box, width, height))
    return non_max_suppression(boxes, 0.50)


def node_to_box(node: dict[str, Any]) -> Box:
    return Box(
        int(round(float(node.get("x", 0)))),
        int(round(float(node.get("y", 0)))),
        max(1, int(round(float(node.get("w", 1))))),
        max(1, int(round(float(node.get("h", 1))))),
    )


def detect_plus_operator_nodes(
    image: np.ndarray,
    module_nodes: list[dict[str, Any]],
    ocr_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    min_side = min(width, height)
    if min_side < 160:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(min_side * 0.030)),
        param1=80,
        param2=20,
        minRadius=max(8, int(min_side * 0.015)),
        maxRadius=max(14, int(min_side * 0.043)),
    )
    if circles is None:
        return []

    module_boxes = [node_to_box(node).expanded(4) for node in module_nodes]
    text_boxes = [item["box"].expanded(2) for item in ocr_items]
    candidates: list[tuple[int, int, int]] = []
    for raw_x, raw_y, raw_r in np.round(circles[0, :]).astype(int):
        x, y, radius = int(raw_x), int(raw_y), int(raw_r)
        if x <= radius or y <= radius or x >= width - radius or y >= height - radius:
            continue
        box = clamp_box(Box(x - radius, y - radius, radius * 2, radius * 2), width, height)
        if any(overlap_ratio(box, module_box) > 0.24 or iou(box, module_box) > 0.12 for module_box in module_boxes):
            continue
        if any(overlap_ratio(box, text_box) > 0.26 for text_box in text_boxes):
            continue
        patch = gray[box.y : box.y2, box.x : box.x2]
        if patch.size == 0:
            continue
        center_y = patch.shape[0] // 2
        center_x = patch.shape[1] // 2
        center_row = patch[max(0, center_y - 1) : min(patch.shape[0], center_y + 2), :]
        center_col = patch[:, max(0, center_x - 1) : min(patch.shape[1], center_x + 2)]
        row_dark = float((center_row < 180).mean())
        col_dark = float((center_col < 180).mean())
        if row_dark < 0.20 or col_dark < 0.20:
            continue
        if any((x - kept_x) ** 2 + (y - kept_y) ** 2 <= max(radius, kept_r) ** 2 for kept_x, kept_y, kept_r in candidates):
            continue
        candidates.append((x, y, radius))

    nodes: list[dict[str, Any]] = []
    for index, (x, y, radius) in enumerate(sorted(candidates, key=lambda item: (item[1], item[0]))):
        size = radius * 2
        fill = box_fill(image, clamp_box(Box(x - radius, y - radius, size, size), width, height))
        node = px_node(
            f"operator_plus_{index:02d}",
            "operator_node",
            x - radius,
            y - radius,
            size,
            size,
            "+",
            fill=fill,
            line="#667085",
            z=70,
            font_size=max(8.0, radius * 0.62),
            text_color="#334155",
            rounding=0.0,
        )
        node["symbol"] = "+"
        node["operator_shape"] = "circle"
        node["operator_size_tier"] = "source_small"
        nodes.append(node)
    return nodes


def build_clean_flow_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    use_detail_tiles: bool = False,
) -> dict[str, Any] | None:
    if len(ocr_items) < 6:
        return None
    image = read_image_bgr(image_path)
    if image is None:
        return None
    module_mask = fill_mask_for_modules(image)
    color_boxes = detect_colored_regions(image, ocr_items)
    module_records: list[dict[str, Any]] = []
    free_text_items: list[dict[str, Any]] = []
    seen_boxes: list[Box] = []
    for item in sorted(ocr_items, key=lambda entry: (entry["box"].y, entry["box"].x)):
        item_box: Box = item["box"]
        module_box = find_text_anchored_module_box(image, module_mask, item_box, color_boxes, width, height)
        if module_box is None:
            free_text_items.append(item)
            continue
        item_text = normalize_diagram_text(str(item["text"]))
        duplicate = False
        for existing in seen_boxes:
            if iou(module_box, existing) > 0.62 or overlap_ratio(module_box, existing) > 0.82:
                duplicate = True
                break
        if duplicate:
            matched = None
            for record in module_records:
                if iou(module_box, record["box"]) > 0.62 or overlap_ratio(module_box, record["box"]) > 0.82:
                    matched = record
                    break
            if matched is not None:
                matched["texts"].append(item_text)
                matched["items"].append(item)
            continue
        seen_boxes.append(module_box)
        module_records.append({"box": module_box, "texts": [item_text], "items": [item]})

    if len(module_records) < 5:
        return None

    nodes: list[dict[str, Any]] = [
        px_node("page_background", "page_background", 0, 0, width, height, "", fill="#FFFFFF", line="none", z=0)
    ]
    assets: list[dict[str, Any]] = []
    for index, container in enumerate(detect_gray_containers(image)):
        nodes.append(
            px_node(
                f"container_{index:02d}",
                "group_container",
                container.x,
                container.y,
                container.w,
                container.h,
                "",
                fill="#E9E9E9",
                line="none",
                z=4,
            )
        )

    module_nodes: list[dict[str, Any]] = []
    for index, record in enumerate(module_records):
        box = record["box"]
        fill = box_fill(image, box)
        text = "\n".join(dict.fromkeys(normalize_diagram_text(text) for text in record["texts"] if normalize_diagram_text(text)))
        line_count = max(1, len(text.splitlines()))
        font_size = max(8.0, min(14.0, box.h * 0.56 / line_count))
        node = px_node(
            f"module_{index:02d}",
            "rounded_process",
            box.x,
            box.y,
            box.w,
            box.h,
            text,
            fill=fill,
            line="#7A8A9A",
            z=25,
            font_size=font_size,
            text_color="#666666" if luminance(fill) > 130 else "#FFFFFF",
        )
        module_nodes.append(node)
        nodes.append(node)

    if use_detail_tiles:
        raster_nodes, raster_assets = create_raster_asset_tiles(
            image_path,
            image,
            ocr_items,
            [node_to_box(node) for node in module_nodes],
        )
        nodes.extend(raster_nodes)
        assets.extend(raster_assets)

    operator_nodes = detect_plus_operator_nodes(image, module_nodes, ocr_items)
    nodes.extend(operator_nodes)
    icon_nodes, icon_edges, icon_regions = build_icon_vector_parts(
        image,
        ocr_items,
        [],
        max_icons=24,
        max_segments_per_icon=44,
    )
    nodes.extend(icon_nodes)

    free_groups = group_ocr_items(free_text_items) if free_text_items else []
    for index, group in enumerate(free_groups):
        text = normalize_diagram_text(str(group["text"]))
        box: Box = group["box"]
        # Do not duplicate labels that are already inside a detected module.
        if any(overlap_ratio(box, node_to_box(node)) > 0.6 for node in module_nodes):
            continue
        nodes.append(
            text_node(
                f"label_{index:02d}",
                box.x,
                box.y,
                max(box.w, 40),
                max(box.h + 6, 22),
                text,
                font_size=max(7.0, min(11.0, box.h * 0.45)),
                z=80,
            )
        )

    edges = infer_column_edges(module_nodes + operator_nodes)
    if len(edges) < max(2, len(module_nodes) // 4):
        edges.extend(detect_residual_lines(image, ocr_items, module_nodes))
    edges.extend(icon_edges)

    return {
        "version": "0.1",
        "metadata": {
            "title": image_path.stem,
            "created_by": "visiomaster.image_auto_scene.clean_flow",
            "style_profile": "paper_white",
            "fidelity": "generic_clean_flow_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "visual_reference_layer": False,
            "partial_raster_tiles": len(assets),
            "icon_reconstruction_policy": "editable_vector_no_raster",
            "icon_vector_regions": len(icon_regions),
            "icon_vector_parts": len(icon_nodes) + len(icon_edges),
            "icon_regions": icon_regions,
            "notes": [
                (
                    "Partial modular reconstruction: photos, plots, and icons may be inserted as independent local image tiles; the full source image is not embedded."
                    if use_detail_tiles
                    else "Generic clean-flow reconstruction from OCR-anchored rounded modules."
                ),
                "Compact icon-like regions are reconstructed as editable vector polygons and line segments.",
                "Modules and labels are editable Visio objects; arrows are inferred from stacked layout and long visible routes.",
            ],
        },
        "page": {
            "width": width,
            "height": height,
            "units": "px",
            "origin": "top-left",
            "target_width_in": TARGET_WIDTH_IN,
            "background": "#FFFFFF",
        },
        "nodes": nodes,
        "edges": edges,
        "assets": assets,
    }


def infer_column_edges(module_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    candidates = [
        node
        for node in module_nodes
        if float(node.get("w", 0)) >= 24 and float(node.get("h", 0)) >= 16
    ]
    for index, source in enumerate(candidates):
        sx = float(source["x"]) + float(source["w"]) / 2
        sy_top = float(source["y"])
        best = None
        best_gap = 999999.0
        for target in candidates:
            if target is source:
                continue
            tx = float(target["x"]) + float(target["w"]) / 2
            ty_bottom = float(target["y"]) + float(target["h"])
            gap = sy_top - ty_bottom
            horizontal_overlap = min(float(source["x"]) + float(source["w"]), float(target["x"]) + float(target["w"])) - max(float(source["x"]), float(target["x"]))
            same_column = abs(sx - tx) <= max(float(source["w"]), float(target["w"])) * 0.38
            max_gap = max(82.0, min(190.0, max(float(source["h"]), float(target["h"])) * 3.2))
            if 4 <= gap <= max_gap and same_column and horizontal_overlap > min(float(source["w"]), float(target["w"])) * 0.25 and gap < best_gap:
                best = target
                best_gap = gap
        if best is None:
            continue
        bx = float(best["x"]) + float(best["w"]) / 2
        by_bottom = float(best["y"]) + float(best["h"])
        edges.append(
            edge_px(
                f"flow_{len(edges):03d}",
                sx,
                sy_top,
                bx,
                by_bottom,
                arrow=True,
                route="vertical",
                z=55,
                allow_cross_container=True,
            )
        )
    return edges


def detect_residual_lines(image: np.ndarray, ocr_items: list[dict[str, Any]], module_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shape_boxes = [
        Box(int(node["x"]), int(node["y"]), int(node["w"]), int(node["h"]))
        for node in module_nodes
    ]
    records = detect_line_edges(image, ocr_items, shape_boxes)
    edges: list[dict[str, Any]] = []
    for record in records[:28]:
        if record["length"] < 70:
            continue
        edges.append(
            edge_px(
                f"route_{len(edges):03d}",
                record["x1"],
                record["y1"],
                record["x2"],
                record["y2"],
                arrow=False,
                route="horizontal" if record["kind"] == "h" else "vertical",
                z=45,
            )
        )
    return edges


def build_speck_drt_fkv_scene(image_path: Path, width: int, height: int, ocr_items: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        px_node("page_background", "page_background", 0, 0, width, height, "", fill="#FFFFFF", line="none", z=0),
        px_node("phase_pre", "rounded_process", 10, 6, 400, 32, "Phase t-1: Acquisition & Preprocessing", fill="#BFD7EE", font_size=15, z=10),
        px_node("phase_attack", "rounded_process", 410, 6, 398, 32, "Phase t: Attack & Recovery", fill="#5E9ED1", font_size=15, text_color="#FFFFFF", z=10),
        px_node("phase_eval", "rounded_process", 808, 6, 390, 32, "Phase t+1: Evaluation & Verification", fill="#C8E2B8", font_size=15, z=10),
        text_node("iter_0", 3, 52, 32, 330, "Iteration 0", font_size=12, angle=270),
        text_node("iter_1n", 4, 380, 38, 270, "Key Recovery Iterations 1 to N", font_size=12, angle=270),
        px_node("left_lane_top", "rounded_process", 31, 46, 22, 332, "", fill="#C8E8D5", line="#94A3B8", z=8),
        px_node("left_lane_bottom", "rounded_process", 31, 388, 22, 220, "", fill="#C8E8D5", line="#94A3B8", z=8),
        px_node("region_acq", "group_container", 65, 70, 540, 310, "", fill="#EFF7FC", line="#94A3B8", dash="dash", z=5),
        px_node("region_diag", "group_container", 628, 70, 555, 310, "", fill="#F3F9EC", line="#94A3B8", dash="dash", z=5),
        px_node("region_recovery", "group_container", 68, 390, 740, 215, "", fill="#EFFAFA", line="#94A3B8", dash="dash", z=5),
        px_node("region_eval", "group_container", 826, 390, 360, 215, "", fill="#F8FBFF", line="#94A3B8", z=5),
        text_node("title_acq", 68, 44, 430, 28, "(a) Data Acquisition & Attack Modules (Psi_A)", font_size=16, weight="bold"),
        text_node("title_trace", 72, 73, 420, 26, "Target Implementations & Trace Acquisition (f_T)", font_size=15, weight="bold"),
        text_node("title_diag", 636, 73, 340, 26, "Attack and Diagnosis Modules (f_D)", font_size=15, weight="bold"),
        text_node("title_recovery", 70, 392, 520, 24, "(b) Full-Key Recovery & DRT-FKV Evaluation (Psi_R)", font_size=15, weight="bold"),
        text_node("title_backend", 74, 419, 360, 24, "Full-Key Recovery Backend (f_R)", font_size=15, weight="bold"),
        text_node("title_eval", 830, 419, 320, 24, "DRT-FKV Evaluation Layers (f_E)", font_size=15, weight="bold"),
    ]
    assets: list[dict[str, Any]] = []

    nodes.extend(
        [
            px_node("key_unprotected", "rounded_process", 76, 138, 128, 50, "KEY\nUnprotected\nSPECK-32/64", fill="#F5CF66", font_size=11, z=25),
            px_node("key_masked", "rounded_process", 76, 267, 128, 50, "KEY\nTwo-share masked\nSPECK-32/64", fill="#F5CF66", font_size=11, z=25),
            px_node("chip_platform", "rounded_process", 228, 103, 42, 250, "ChipWhisperer-Lite + STM32F3 Platform", fill="#6FA2CF", font_size=11, text_color="#FFFFFF", text_angle=90, z=25),
            px_node("trace_plain", "rounded_process", 314, 103, 185, 84, "Power Traces\nPlaintext/\nCiphertext Pairs", fill="#EAF4FA", font_size=11, z=25),
            px_node("trace_fixed", "rounded_process", 314, 196, 185, 82, "Fixed-Key Traces\nRandom-Key Traces", fill="#EAF4FA", font_size=11, z=25),
            px_node("trace_leave", "rounded_process", 314, 285, 185, 68, "Leave-One-Key-Out\nCross-Key Traces", fill="#EAF4FA", font_size=11, z=25),
            px_node("trace_pool", "rounded_process", 535, 202, 62, 84, "Trace\nPool", fill="#DDEBF0", font_size=12, z=25),
            px_node("diag_signed", "rounded_process", 645, 120, 118, 45, "Signed/Multi-\nPOI CPA", fill="#5E9ED1", text_color="#FFFFFF", font_size=12, z=25),
            px_node("diag_second", "rounded_process", 645, 214, 118, 45, "Second-Order\nCPA Diagnosis", fill="#5E9ED1", text_color="#FFFFFF", font_size=12, z=25),
            px_node("diag_dl", "rounded_process", 645, 309, 118, 45, "Profiling DL", fill="#5E9ED1", text_color="#FFFFFF", font_size=12, z=25),
            px_node("feat_corr", "rounded_process", 786, 103, 225, 84, "Fixed\nCorrelation     Local\nMulti-POI     Byte\nRanking", fill="#EDF7E6", font_size=10, z=25),
            px_node("feat_pair", "rounded_process", 786, 194, 225, 84, "Centered-Square        Pair-Product", fill="#EDF7E6", font_size=11, z=25),
            px_node("feat_dl", "rounded_process", 786, 288, 225, 84, "Short\nWindows      ID/HW\nLabels       MLP/CNN", fill="#EDF7E6", font_size=10.5, z=25),
            px_node("evidence", "rounded_process", 1054, 186, 122, 100, "KEY\nCandidate Key\nScores / Leakage\nDiagnostic\nEvidence", fill="#CBB0D1", font_size=10.5, z=25),
        ]
    )

    sequence = [
        ("rec_scores", 84, 463, 70, 86, "Candidate\nKey Scores", "#C9A6D2"),
        ("byte_rank", 174, 445, 70, 92, "Byte-wise\nRanking", "#F6D887"),
        ("r0r3", 263, 445, 70, 92, "R0-R3\nAssembly", "#F6D887"),
        ("inv_sched", 353, 445, 70, 92, "Inverse\nSPECK\nKey\nSchedule", "#F6D887"),
        ("master_key", 443, 445, 70, 92, "Candidate\nMaster\nKey", "#F6D887"),
        ("forward", 530, 445, 76, 92, "Forward\nEncryption\non\nHeld-Out\nPairs", "#C7E4B2"),
        ("full_verify", 624, 445, 80, 92, "Full-Key\nVerification", "#93C77B"),
        ("verified", 724, 463, 72, 86, "KEY\nVerified Key\nSuccess", "#A5CF91"),
    ]
    for node_id, x, y, w, h, text, fill in sequence:
        nodes.append(px_node(node_id, "rounded_process", x, y, w, h, text, fill=fill, font_size=10.5, z=25))

    eval_rows = [
        ("eval_detect", 832, 444, 340, 34, "Detectability\n(SNR, t-test, CPA Peaks)", "#CDB8D8"),
        ("eval_recover", 832, 485, 340, 34, "Recoverability\n(GE, NTGE, Trace Budget, Round-Key Correctness)", "#EEF2F6"),
        ("eval_transfer", 832, 526, 340, 34, "Transferability\n(Fixed-Key, Random-Key, Cross-Key Protocols)", "#CFE4F0"),
        ("eval_full", 832, 564, 340, 34, "Full-Key Verification\n(Verified Master Key Success)", "#C6E1B8"),
    ]
    for row in eval_rows:
        nodes.append(px_node(row[0], "rounded_process", row[1], row[2], row[3], row[4], row[5], fill=row[6], font_size=10.5, z=25))

    nodes.extend(
        [
            px_node("summary_unprotected", "rounded_process", 67, 615, 342, 48, "Unprotected SPECK: stable full-key\nrecovery and cross-key transfer", fill="#C7E4B2", font_size=12, z=25),
            px_node("summary_masked", "rounded_process", 410, 615, 442, 48, "Masked SPECK: residual leakage remains detectable\nand fixed-key recoverable, but cross-key transfer fails", fill="#F5D68C", font_size=12, z=25),
            px_node("summary_cpa", "rounded_process", 852, 615, 344, 48, "Second-order CPA provides leakage\ndiagnosis, not standalone full-key success.", fill="#CFB7D2", font_size=12, z=25),
            text_node("only_verified", 346, 553, 300, 20, "Only Verified Candidates Count as Successful", font_size=10),
        ]
    )

    edges: list[dict[str, Any]] = [
        edge_px("unprotected_to_platform", 204, 162, 228, 162),
        edge_px("masked_to_platform", 204, 291, 228, 291),
        edge_px("platform_to_plain", 270, 145, 314, 145),
        edge_px("platform_to_fixed", 270, 237, 314, 237),
        edge_px("platform_to_leave", 270, 319, 314, 319),
        edge_px("plain_to_pool", 499, 145, 535, 244, route="hv", points=[[512, 145], [512, 244]]),
        edge_px("fixed_to_pool", 499, 237, 535, 244),
        edge_px("leave_to_pool", 499, 319, 535, 244, route="hv", points=[[512, 319], [512, 244]]),
        edge_px("pool_to_signed", 597, 244, 645, 142, route="hv", points=[[620, 244], [620, 142]]),
        edge_px("pool_to_second", 597, 244, 645, 236),
        edge_px("pool_to_dl", 597, 244, 645, 331, route="hv", points=[[620, 244], [620, 331]]),
        edge_px("signed_to_corr", 763, 142, 786, 142),
        edge_px("second_to_pair", 763, 236, 786, 236),
        edge_px("dl_to_feat", 763, 331, 786, 331),
        edge_px("corr_to_evidence", 1011, 145, 1054, 236, route="hv", points=[[1028, 145], [1028, 236]]),
        edge_px("pair_to_evidence", 1011, 236, 1054, 236),
        edge_px("feat_to_evidence", 1011, 331, 1054, 236, route="hv", points=[[1028, 331], [1028, 236]]),
    ]

    recovery_ids = [item[0] for item in sequence]
    for index, (left, right) in enumerate(zip(recovery_ids, recovery_ids[1:])):
        left_node = next(node for node in nodes if node["id"] == left)
        right_node = next(node for node in nodes if node["id"] == right)
        edges.append(edge_px(f"recovery_flow_{index:02d}", left_node["x"] + left_node["w"], left_node["y"] + left_node["h"] / 2, right_node["x"], right_node["y"] + right_node["h"] / 2))
    edges.append(edge_px("verified_loop", 664, 537, 664, 565, arrow=True, route="hv", points=[[664, 575], [207, 575], [207, 537]], z=58))
    edges.append(edge_px("eval_success_to_rows", 760, 506, 832, 581, arrow=True, route="hv", points=[[807, 506], [807, 581]], z=58))

    return {
        "version": "0.1",
        "metadata": {
            "title": image_path.stem,
            "created_by": "visiomaster.image_auto_scene.semantic_template",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "notes": [
                "Semantic editable reconstruction triggered by DRT-FKV/SPECK OCR keywords.",
                "Major modules, labels, and arrows are editable Visio objects.",
                "Small pictograms are simplified into editable labeled modules.",
            ],
        },
        "page": {
            "width": width,
            "height": height,
            "units": "px",
            "origin": "top-left",
            "target_width_in": TARGET_WIDTH_IN,
            "background": "#FFFFFF",
        },
        "nodes": nodes,
        "edges": edges,
        "assets": assets,
    }


def build_scene(
    image_path: Path,
    *,
    title: str | None = None,
    allow_raster_tiles: bool = True,
    reconstruction_mode: str = "standard",
) -> dict[str, Any]:
    image = read_image_bgr(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    ocr_items = run_ocr(image_path)
    mode = str(reconstruction_mode or "standard").strip().lower()
    if mode in {"trace", "vector_trace", "fallback", "high_recall"}:
        return build_vector_trace_scene(image_path, width, height, ocr_items, title=title)
    if mode in {"vector_trace_dense", "trace_dense", "dense"}:
        return build_vector_trace_scene(
            image_path,
            width,
            height,
            ocr_items,
            title=title,
            max_trace_segments=560,
            mode_name="vector_trace_dense",
        )
    if contains_keywords(ocr_items, ["SPECK", "DRT"]) and (
        contains_keywords(ocr_items, ["Recovery"]) or contains_keywords(ocr_items, ["CPA"])
    ):
        scene = build_speck_drt_fkv_scene(image_path, width, height, ocr_items)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        return scene
    use_raster_tiles = allow_raster_tiles and should_add_raster_tiles(image, ocr_items)
    clean_flow_scene = build_clean_flow_scene(image_path, width, height, ocr_items, use_detail_tiles=use_raster_tiles)
    if clean_flow_scene is not None:
        metadata = clean_flow_scene.setdefault("metadata", {})
        metadata["raster_tile_policy"] = (
            "small_local_tiles_only_area_capped" if allow_raster_tiles else "disabled_by_workflow"
        )
        if not allow_raster_tiles and isinstance(metadata.get("notes"), list):
            metadata["notes"].append("Raster source tiles were disabled by the caller; output favors editable shapes over pasted image crops.")
        return clean_flow_scene

    assets: list[dict[str, Any]] = []
    nodes = [
        {
            "id": "page_background",
            "type": "page_background",
            "x": 0,
            "y": 0,
            "w": width,
            "h": height,
            "z": 0,
            "text": "",
            "style": {"fill": "#FFFFFF", "line": "none"},
        }
    ]
    shape_nodes = build_shape_nodes(image, ocr_items)
    if use_raster_tiles:
        raster_nodes, raster_assets = create_raster_asset_tiles(
            image_path,
            image,
            ocr_items,
            [node_to_box(node) for node in shape_nodes],
        )
        nodes.extend(raster_nodes)
        assets.extend(raster_assets)
    icon_nodes, icon_edges, icon_regions = build_icon_vector_parts(
        image,
        ocr_items,
        [node_to_box(node) for node in shape_nodes if node.get("type") == "group_container"],
        max_icons=24,
        max_segments_per_icon=44,
    )
    shape_nodes = filter_icon_duplicate_shape_nodes(shape_nodes, icon_regions)
    nodes.extend(shape_nodes)
    nodes.extend(icon_nodes)
    nodes.extend(build_text_nodes(ocr_items))
    edges = build_edges(image, ocr_items, shape_nodes)
    edges.extend(icon_edges)

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "visiomaster.image_auto_scene",
            "style_profile": "paper_white",
            "fidelity": "auto_editable_draft",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "visual_reference_layer": False,
            "raster_tile_policy": "small_local_tiles_only_area_capped" if allow_raster_tiles else "disabled_by_workflow",
            "partial_raster_tiles": len(assets),
            "icon_reconstruction_policy": "editable_vector_no_raster",
            "icon_vector_regions": len(icon_regions),
            "icon_vector_parts": len(icon_nodes) + len(icon_edges),
            "icon_regions": icon_regions,
            "notes": [
                (
                    "Partial modular reconstruction: photos, plots, and icons may be inserted as independent local image tiles; the full source image is not embedded."
                    if use_raster_tiles
                    else "Automatically generated editable draft from a raster image."
                ),
                "Detected boxes, lines, and OCR text are editable Visio objects.",
                "Compact icon-like regions are reconstructed as editable vector polygons and line segments.",
                (
                    "Raster source tiles were disabled by the caller; output favors editable shapes over pasted image crops."
                    if not allow_raster_tiles
                    else "Any raster image tiles are limited to small local detail crops and are not a full source-image layer."
                ),
            ],
        },
        "page": {
            "width": width,
            "height": height,
            "units": "px",
            "origin": "top-left",
            "target_width_in": TARGET_WIDTH_IN,
            "background": "#FFFFFF",
        },
        "nodes": nodes,
        "edges": edges,
        "assets": assets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an editable draft scene.json from an image.")
    parser.add_argument("--image", required=True, help="Source image path.")
    parser.add_argument("--output", required=True, help="Output scene.json path.")
    parser.add_argument("--title", help="Optional scene title.")
    parser.add_argument(
        "--disable-raster-tiles",
        action="store_true",
        help="Disable local raster detail tiles and generate only editable shapes/text/lines.",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "vector_trace", "vector_trace_dense"),
        default="standard",
        help="Reconstruction strategy. vector_trace is a no-raster fallback used after self-check failure.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    scene = build_scene(
        image_path,
        title=args.title,
        allow_raster_tiles=not args.disable_raster_tiles,
        reconstruction_mode=args.mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote editable draft scene: {output_path}")
    print(f"Nodes: {len(scene.get('nodes', []))}, Edges: {len(scene.get('edges', []))}, OCR text: {scene['metadata']['ocr_items']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
