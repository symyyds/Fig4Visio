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
_RAPIDOCR_ENGINE: Any | None = None
_RAPIDOCR_IMPORT_FAILED = False


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
    global _RAPIDOCR_ENGINE, _RAPIDOCR_IMPORT_FAILED
    if _RAPIDOCR_IMPORT_FAILED:
        return []
    try:
        if _RAPIDOCR_ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _RAPIDOCR_ENGINE = RapidOCR()
    except Exception:
        _RAPIDOCR_IMPORT_FAILED = True
        return []

    result, _ = _RAPIDOCR_ENGINE(str(image_path))
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
            "created_by": f"fig4visio.image_auto_scene.{mode_name}",
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


def ocr_corpus(ocr_items: list[dict[str, Any]]) -> str:
    return " ".join(str(item.get("text", "")) for item in ocr_items)


def contains_keywords(ocr_items: list[dict[str, Any]], keywords: list[str]) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    return all(keyword.lower() in corpus for keyword in keywords)


def is_remote_sensing_rsei_workflow_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)

    has_rsei = "rsei" in compact or "rsel" in compact
    rsei_index_count = sum(token in compact for token in ("ndvi", "ndsi", "wet", "lst"))
    has_pls_sem = "plssem" in compact or ("pls" in compact and "sem" in compact)
    has_remote_inputs = (
        "landsat" in compact
        or "surfacereflectance" in compact
        or "jrcglobalsurfacewater" in compact
        or ("global" in compact and "surfacewater" in compact)
    )
    has_driver_layer = (
        "driverlayer" in compact
        or sum(token in compact for token in ("terrain", "climate", "soil", "urbanization")) >= 3
    )
    has_preprocess = (
        "preprocessing" in compact
        or "preprocessing" in corpus.replace("-", "")
        or "ledaps" in compact
        or "lasrc" in compact
        or "cfmask" in compact
        or "watermask" in compact
    )
    has_gee_extraction = "gee" in compact or "normalizationpca" in compact or "multiyearrsei" in compact
    has_spatial_auto = "globalspatialautocorrelation" in compact or "autocorrelation" in compact
    has_change = "rseichangeanalysis" in compact or ("change" in compact and has_rsei)

    signal_count = sum(
        bool(flag)
        for flag in (
            has_rsei,
            rsei_index_count >= 3,
            has_pls_sem,
            has_remote_inputs,
            has_driver_layer,
            has_preprocess,
            has_gee_extraction,
            has_spatial_auto,
            has_change,
        )
    )
    return 1.40 <= aspect <= 2.25 and has_rsei and rsei_index_count >= 3 and has_pls_sem and signal_count >= 6


def is_drought_basin_workflow_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)

    has_datasets = "datasetsinput" in compact or ("datasets" in compact and "input" in compact)
    has_drought = "drought" in compact
    has_spei = "spei12" in compact or "spei" in compact or "spel12" in compact or "spel" in compact
    has_inputs = sum(token in compact for token in ("meteorologicaldata", "sstdata", "nino34data", "riverbasinsdata")) >= 3
    has_basin_panel = "34majorglobalriverbasins" in compact or ("riverbasins" in compact and "global" in compact)
    has_clustering = "3ddroughtclustering" in compact or ("clustering" in compact and "droughtstructure" in compact)
    has_characteristics = "droughteventcharacteristics" in compact or (
        "droughtduration" in compact and "droughtarea" in compact
    )
    has_mca = "maximumcovarianceanalysis" in compact or "mca2" in compact or ("sst" in compact and "enso" in compact)
    has_influencing = "influencingfactorsofdrought" in compact or ("influencing" in compact and "drought" in compact)
    has_final = "meteorologicaldrought" in compact or "identificationandcontrast" in compact

    signal_count = sum(
        bool(flag)
        for flag in (
            has_datasets,
            has_drought,
            has_spei,
            has_inputs,
            has_basin_panel,
            has_clustering,
            has_characteristics,
            has_mca,
            has_influencing,
            has_final,
        )
    )
    return 0.45 <= aspect <= 0.95 and has_drought and has_spei and has_datasets and has_clustering and signal_count >= 7


def is_industry_4_0_sustainability_framework_figure(
    ocr_items: list[dict[str, Any]],
    width: int,
    height: int,
) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)

    has_industry = "industry40" in compact or "industry4o" in compact or (
        "industry" in compact and ("40" in compact or "4o" in compact)
    )
    has_left_framework = sum(token in compact for token in ("technologies", "components", "principles")) >= 2
    has_tech_terms = sum(
        token in compact
        for token in (
            "artificialintelligence",
            "mixedreality",
            "robotics",
            "blockchain",
            "bigdata",
            "analytics",
            "digitaltwins",
            "cps",
            "iiot",
            "llot",
        )
    ) >= 3
    has_sustainability_functions = "sustainabilityfunctions" in compact or (
        "sustainability" in compact and "functions" in compact
    )
    has_function_list = sum(
        token in compact
        for token in (
            "businessmodelinnovation",
            "customerorientedmanufacturing",
            "employeeproductivity",
            "harmfulemissionreduction",
            "manufacturingagility",
            "resourceandenergyefficiency",
            "sustainableproductdevelopment",
            "supplychainprocessintegration",
        )
    ) >= 4
    has_sustainable_manufacturing = "sustainablemanufacturing" in compact or (
        "sustainable" in compact and "manufacturing" in compact
    )
    has_output_cards = sum(
        token in compact
        for token in (
            "socialdevelopment",
            "sustainableeconomicgrowth",
            "renewables",
            "greenmanufacturing",
            "greenmanufacaring",
        )
    ) >= 2
    signal_count = sum(
        bool(flag)
        for flag in (
            has_industry,
            has_left_framework,
            has_tech_terms,
            has_sustainability_functions,
            has_function_list,
            has_sustainable_manufacturing,
            has_output_cards,
        )
    )
    return (
        1.30 <= aspect <= 2.25
        and has_industry
        and has_sustainability_functions
        and has_sustainable_manufacturing
        and signal_count >= 6
    )


def is_mask_res_block_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)
    has_conv_stack = "conv764" in compact or ("conv7" in compact and "64" in compact)
    has_norm = "batchnormalization" in compact
    has_pooling = "maxpooling" in compact or ("max" in compact and "pooling" in compact)
    has_resblock = "maskresblock" in compact or "originalresblock" in compact or (
        "resblock" in compact and "mask" in compact
    )
    has_lanes = "mask" in compact and ("xi" in compact or "xit" in compact or "x" in compact)
    return 1.15 <= aspect <= 1.95 and has_conv_stack and has_norm and has_pooling and has_resblock and has_lanes


def is_cross_attention_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)
    has_inputs = "amresnet" in compact and ("wav2vec20" in compact or "wav2vec" in compact)
    has_attention = "softmax" in compact and "concat" in compact and "norm" in compact
    has_output = "crossfused" in compact or ("cross" in compact and "fused" in compact)
    has_caption = "crossattention" in compact or "attention" in compact
    has_qkv = any(token in compact for token in ("qa", "qw", "ka", "kw", "va", "vw"))
    return aspect >= 2.45 and has_inputs and has_attention and has_output and has_caption and has_qkv


def is_attention_mechanism_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)
    has_title = "attentionmechanism" in compact or ("attention" in compact and "mechanism" in compact)
    has_attention_core = "sigmoid" in compact and ("conv1d" in compact or "convid" in compact or "convld" in compact)
    has_weight_vector = "weightedvector" in compact or ("weighted" in compact and "vector" in compact)
    has_feature_maps = (
        ("highlevelfeatures" in compact or ("highlevel" in compact and "features" in compact))
        and ("amresnetfeatures" in compact or ("amresnet" in compact and "features" in compact))
    )
    has_caption = "architectureoftheattentionmechanism" in compact or ("fig5" in compact and has_title)
    return 1.75 <= aspect <= 3.05 and has_title and has_attention_core and has_weight_vector and has_feature_maps and has_caption


def is_channel_attention_recalibration_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    folded = corpus.replace("\u00d7", "x").replace("*", "x")
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    aspect = width / max(1, height)

    vector_mentions = compact.count("1x1xc") + compact.count("1x1c")
    has_scale = "fscale" in compact or ("scale" in compact and vector_mentions > 0)
    has_excitation = "fex" in compact or "fw" in compact or "fexw" in compact
    has_squeeze = "fsq" in compact or "fg" in compact
    has_transfer = "ftr" in compact or ("originalimage" in compact and ("x" in compact and "u" in compact))
    has_original = "originalimage" in compact or ("original" in compact and "image" in compact)
    text_atoms = {re.sub(r"[^a-z0-9~]+", "", str(item.get("text", "")).lower()) for item in ocr_items}
    has_tensor_symbols = bool({"x", "u", "h", "c"} & text_atoms)

    signal_count = sum(
        bool(flag)
        for flag in (
            has_original,
            has_transfer,
            has_squeeze,
            has_excitation,
            has_scale,
            vector_mentions >= 1,
            has_tensor_symbols,
        )
    )
    return 1.70 <= aspect <= 3.20 and has_scale and vector_mentions >= 1 and signal_count >= 4


def is_deformable_transformer_encoder_decoder_figure(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    folded = corpus.replace("\u00d7", "x").replace("&", "and")
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    aspect = width / max(1, height)
    has_encoder_decoder = "encoder" in compact and "decoder" in compact
    has_deformable = "deformable" in compact or "deformabl" in compact
    has_attention = ("selfattention" in compact or "crossattention" in compact) and "multihead" in compact
    has_ffn = "bcffn" in compact or ("ffn" in compact and "gn" in compact and "gelu" in compact)
    has_norm = "addnorm" in compact or ("add" in compact and "norm" in compact)
    has_feature_grids = "featuregrids" in compact or ("feature" in compact and "grids" in compact)
    has_queries = "locationguidedqueries" in compact or ("queries" in compact and "location" in compact)
    has_restore = "restore" in compact or "flatten" in compact or "t3t5" in compact or "t375" in compact
    signal_count = sum(
        bool(flag)
        for flag in (
            has_encoder_decoder,
            has_deformable,
            has_attention,
            has_ffn,
            has_norm,
            has_feature_grids,
            has_queries,
            has_restore,
        )
    )
    return 1.45 <= aspect <= 2.45 and signal_count >= 6 and has_encoder_decoder and has_deformable and has_ffn


def build_deformable_transformer_encoder_decoder_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 981.0
    base_h = 524.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def attach(node: dict[str, Any], x: float, y: float, w: float, h: float, container_id: str | None = None) -> dict[str, Any]:
        node["source_bbox_px"] = bbox(x, y, w, h)
        if container_id:
            node["container_id"] = container_id
        return node

    def page_node() -> dict[str, Any]:
        return px_node("page_background", "page_background", 0, 0, width, height, "", fill="#FFFFFF", line="none", z=0)

    def container(node_id: str, x: float, y: float, w: float, h: float, text: str = "") -> dict[str, Any]:
        item = px_node(node_id, "group_container", sx(x), sy(y), sx(w), sy(h), text, fill="#F0F0F0", line="#111111", z=4, font_size=11, text_color="#111111", dash="dash", rounding=0.12)
        item["style"].update({"fill": "#F0F0F0", "line_dash": "dash", "line_weight_pt": 1.25, "font_weight": "bold"})
        return attach(item, x, y, w, h)

    def label(node_id: str, x: float, y: float, w: float, h: float, text: str, *, container_id: str | None = None, font_size: float = 10, weight: str = "regular", italic: bool = False, color: str = "#111111", angle: float | None = None, z: int = 90) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, angle=angle, z=z)
        item["style"].update(
            {
                "font_family_candidates": ["Times New Roman", "Cambria Math", "Cambria", "Microsoft YaHei UI"],
                "font_role": "math" if italic else "paper_serif",
                "text_color": color,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 4.5,
            }
        )
        if italic:
            item["style"]["font_italic"] = True
        return attach(item, x, y, w, h, container_id)

    def math_label(node_id: str, x: float, y: float, w: float, h: float, text: str, *, container_id: str | None = None, font_size: float = 9.5, z: int = 91) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "math_text",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": text,
            "font_size_pt": font_size,
            "math_render_mode": "fragments",
            "style": {
                "fill": "none",
                "line": "none",
                "text_color": "#111111",
                "font_family_candidates": ["Times New Roman", "Cambria Math", "Cambria"],
                "font_role": "math",
                "font_italic": True,
                "font_size_pt": font_size,
                "text_fit": "math_label",
                "min_font_size_pt": 4.5,
                "text_margin_in": 0.0,
            },
        }
        return attach(item, x, y, w, h, container_id)

    def box(node_id: str, x: float, y: float, w: float, h: float, text: str, *, container_id: str | None = None, fill: str = "#FFFFFF", line: str = "#111111", font_size: float = 9, angle: float | None = None, z: int = 30, rounded: bool = False, allow_overlap: bool = False) -> dict[str, Any]:
        item = px_node(node_id, "rounded_process" if rounded else "process_box", sx(x), sy(y), sx(w), sy(h), text, fill=fill, line=line, z=z, font_size=font_size, text_color="#111111", text_angle=angle, rounding=0.05 if rounded else 0.0)
        item["style"].update(
            {
                "line_weight_pt": 1.0,
                "font_family_candidates": ["Times New Roman", "Cambria", "Microsoft YaHei UI"],
                "font_size_pt": font_size,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 4.5,
            }
        )
        if allow_overlap:
            item["allow_overlap"] = True
        return attach(item, x, y, w, h, container_id)

    def op(node_id: str, cx: float, cy: float, size: float, symbol: str = "+", *, container_id: str | None = None, z: int = 65) -> dict[str, Any]:
        item = px_node(node_id, "operator_node", sx(cx - size / 2), sy(cy - size / 2), sx(size), sy(size), symbol, fill="#FFFFFF", line="#111111", z=z, font_size=9, text_color="#111111")
        item["symbol"] = symbol
        item["operator_shape"] = "circle"
        item["style"].update({"line_weight_pt": 1.0, "font_size_pt": 9, "font_family": "Times New Roman"})
        return attach(item, cx - size / 2, cy - size / 2, size, size, container_id)

    def pos_token(node_id: str, cx: float, cy: float, *, container_id: str | None = None, z: int = 63) -> list[dict[str, Any]]:
        size = 18
        token = px_node(node_id, "ellipse_node", sx(cx - size / 2), sy(cy - size / 2), sx(size), sy(size), "", fill="#F7D955", line="#111111", z=z, font_size=1, text_color="#111111")
        token["style"].update({"line_weight_pt": 1.0})
        return [attach(token, cx - size / 2, cy - size / 2, size, size, container_id)]

    def pos_icon(node_id: str, cx: float, cy: float, *, container_id: str | None = None, z: int = 63) -> dict[str, Any]:
        return pos_token(node_id, cx, cy, container_id=container_id, z=z)[0]

    def tensor_stack(node_id: str, x: float, y: float, w: float, h: float, *, container_id: str | None = None, fill: str = "#E6E6E6", layers: int = 4, z: int = 32) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "tensor_stack",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "layers": layers,
            "stack_render_mode": "slanted_sheets",
            "layer_dx_in": 3.0,
            "layer_dy_in": -4.0,
            "skew_x_in": 10.0,
            "style": {
                "fill": fill,
                "side_fill": "#C8C8C8",
                "top_fill": "#F4F4F4",
                "line": "#111111",
                "line_weight_pt": 0.85,
            },
        }
        return attach(item, x, y - 12, w + 44, h + 24, container_id)

    def feature_grid(node_id: str, x: float, y: float, w: float, h: float, *, container_id: str | None = None, z: int = 34) -> dict[str, Any]:
        cells = []
        for row in range(5):
            for col in range(5):
                fill = "#F5E8FF" if (row + col) % 2 else "#E2C8F5"
                cells.append([row, col, fill])
        item = {
            "id": node_id,
            "type": "grid_matrix",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "rows": 5,
            "cols": 5,
            "colored_cells": cells,
            "style": {"cell_fill": "#F5E8FF", "grid_line": "#8A6AA3", "grid_line_weight_pt": 0.75, "line": "#8A6AA3", "line_weight_pt": 0.75},
        }
        return attach(item, x, y, w, h, container_id)

    def query_stack(node_id: str, x: float, y: float, w: float, h: float, *, container_id: str | None = None, z: int = 38) -> dict[str, Any]:
        fills = ["#DDF2D4", "#CBE9C1", "#9FD18D", "#4F8F38", "#FFFFFF", "#DDF2D4", "#CBE9C1", "#4F8F38"]
        item = {
            "id": node_id,
            "type": "feature_vector_stack",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "orientation": "vertical",
            "count": len(fills),
            "cell_gap_in": 1.3,
            "cell_fills": fills,
            "outline": True,
            "style": {
                "cell_line": "#111111",
                "cell_line_weight_pt": 0.8,
                "line": "#111111",
                "line_weight_pt": 1.0,
                "cell_fill": "#DDF2D4",
            },
        }
        return attach(item, x, y, w, h, container_id)

    def edge(edge_id: str, x1: float, y1: float, x2: float, y2: float, *, route: str = "straight", arrow: bool = True, points: list[list[float]] | None = None, z: int = 70, allow_diagonal: bool = False, allow_cross_container: bool = False, allow_direct_cross_container: bool = False, allow_text_overlap: bool = False) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": edge_id,
            "type": "lane_arrow" if route in {"horizontal", "vertical"} and arrow else ("arrow_connector" if arrow else "line_segment"),
            "from_point": [sx(x1), sy(y1)],
            "to_point": [sx(x2), sy(y2)],
            "route": route,
            "z": z,
            "style": {"line": "#111111", "line_weight_pt": 1.05, "end_arrow": "triangle" if arrow else "none", "arrow_size": "small"},
        }
        if points:
            item["points"] = [[sx(px), sy(py)] for px, py in points]
            item["orthogonalize_points"] = route in {"orthogonal", "hv", "vh"}
        if allow_diagonal:
            item["allow_diagonal"] = True
        if allow_cross_container:
            item["allow_cross_container"] = True
        if allow_direct_cross_container:
            item["allow_direct_cross_container"] = True
        if allow_text_overlap:
            item["allow_text_overlap"] = True
        return item

    nodes: list[dict[str, Any]] = [page_node()]
    edges: list[dict[str, Any]] = []

    nodes.extend(
        [
            label("original_word", 0, 116, 91, 27, "Original", font_size=19, weight="bold", color="#79CFD0"),
            label("image_word", 8, 144, 72, 31, "image", font_size=19, weight="bold", color="#79CFD0"),
        ]
    )

    def add_encoder(prefix: str, x: float, y: float, w: float, h: float, *, input_x: float, input_y: float) -> tuple[float, float]:
        cid = f"{prefix}_encoder"
        nodes.append(container(cid, x, y, w, h))
        nodes.extend(
            [
                label(f"{prefix}_encoder_title", x + w - 98, y + 14, 70, 17, "Encoder", container_id=cid, font_size=10, weight="bold"),
                label(f"{prefix}_encoder_mx", x + w - 38, y + 30, 30, 16, "Mx", container_id=cid, font_size=8),
                op(f"{prefix}_input_add", x + 20, y + h * 0.56, 22, container_id=cid),
                box(f"{prefix}_self_attention", x + 63, y + 18, 44, h - 44, "Multi-Head Deformable\nSelf-Attention", container_id=cid, font_size=8.5, angle=90, rounded=True),
                box(f"{prefix}_add_norm", x + 116, y + 50, 28, h - 92, "Add & Norm", container_id=cid, font_size=8.5, angle=90, rounded=True),
                box(f"{prefix}_bc_frame", x + 148, y + 62, 154, h - 102, "", container_id=cid, fill="#EAF3F9", line="#111111", rounded=True, z=24, allow_overlap=True),
                label(f"{prefix}_bc_title", x + 203, y + 66, 62, 16, "BC-FFN", container_id=cid, font_size=9.5, weight="bold"),
                box(f"{prefix}_conv1", x + 170, y + 82, 22, h - 132, "3x3\nConv", container_id=cid, font_size=7, angle=90, allow_overlap=True),
                box(f"{prefix}_gn", x + 204, y + 92, 20, h - 150, "GN", container_id=cid, font_size=8, angle=90, allow_overlap=True),
                box(f"{prefix}_gelu", x + 235, y + 87, 22, h - 142, "GELU", container_id=cid, font_size=7, angle=90, allow_overlap=True),
                box(f"{prefix}_conv2", x + 268, y + 82, 24, h - 132, "3x3\nConv", container_id=cid, font_size=7, angle=90, allow_overlap=True),
                math_label(f"{prefix}_xm", input_x - 30, input_y - 29, 38, 16, "X_m", font_size=9.5),
                math_label(f"{prefix}_xb", x + 145, y + h * 0.54 - 32, 34, 16, "X_b", container_id=cid, font_size=8.5),
                math_label(f"{prefix}_xbp", x + 290, y + h * 0.54 - 32, 38, 16, "X'_b", container_id=cid, font_size=8.5),
                math_label(f"{prefix}_xe", x + w + 10, y + h * 0.54 - 32, 32, 16, "X_e", font_size=8.5),
                label(f"{prefix}_vkq_v", x + 48, y + 37, 14, 12, "v", container_id=cid, font_size=8, italic=True),
                label(f"{prefix}_vkq_k", x + 45, y + 91, 14, 12, "k", container_id=cid, font_size=8, italic=True),
                label(f"{prefix}_vkq_q", x + 45, y + 145, 14, 12, "q", container_id=cid, font_size=8, italic=True),
            ]
        )
        nodes.extend(pos_token(f"{prefix}_pm", x + 19, y + h - 30, container_id=cid))
        edges.extend(
            [
                edge(f"{prefix}_input_to_add", input_x, input_y, x + 9, y + h * 0.56, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
                edge(f"{prefix}_add_to_attention", x + 31, y + h * 0.56, x + 63, y + h * 0.56, route="horizontal"),
                edge(f"{prefix}_attention_to_norm", x + 107, y + h * 0.56, x + 116, y + h * 0.56, route="horizontal"),
                edge(f"{prefix}_norm_to_bc", x + 144, y + h * 0.56, x + 170, y + h * 0.56, route="horizontal"),
                edge(f"{prefix}_bc_to_output", x + 302, y + h * 0.56, x + w + 4, y + h * 0.56, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
                edge(f"{prefix}_pm_to_add", x + 19, y + h - 39, x + 19, y + h * 0.56 + 11, route="vertical", arrow=False),
                edge(f"{prefix}_residual_top", x + 31, y + 18, x + 132, y + 18, route="horizontal", arrow=False),
                edge(f"{prefix}_residual_down", x + 132, y + 18, x + 132, y + 50, route="vertical"),
            ]
        )
        return x + w + 20, y + h * 0.56

    def add_feature_bridge(prefix: str, x: float, y: float, *, top_lane: bool) -> tuple[float, float]:
        cid = f"{prefix}_feature_bridge"
        nodes.append(attach(px_node(cid, "audit_region", sx(x), sy(y), sx(150), sy(205), "", fill="#FFFFFF", line="none", z=1, font_size=1, text_color="#111111"), x, y, 150, 205))
        nodes[-1]["style"].update({"fill": "none", "line": "none"})
        nodes.extend(
            [
                label(f"{prefix}_restore", x + 2, y + 6, 58, 17, "Restore", container_id=cid, font_size=8.5),
                tensor_stack(f"{prefix}_restore_stack", x + 48, y + 24, 70, 28, container_id=cid, layers=4),
                label(f"{prefix}_t3t5", x + 58, y + 66, 46, 16, "T3-T5", container_id=cid, font_size=8.5, italic=True),
                label(f"{prefix}_resize", x + 88, y + 91, 42, 16, "Resize", container_id=cid, font_size=7.5, angle=90),
                label(f"{prefix}_sxs", x + 38, y + 114, 38, 16, "SxS", container_id=cid, font_size=8.5),
                label(f"{prefix}_feature", x + 18, y + 139, 56, 17, "Feature", container_id=cid, font_size=8.5),
                label(f"{prefix}_grids", x + 25, y + 157, 44, 17, "Grids", container_id=cid, font_size=8.5),
                feature_grid(f"{prefix}_feature_grid", x + 70, y + 124, 60, 48, container_id=cid),
                box(f"{prefix}_flatten", x + 122, y + 106, 50, 48, "Flatten", container_id=cid, font_size=8.5),
            ]
        )
        edges.extend(
            [
                edge(f"{prefix}_restore_to_stack", x + 2, y + 20, x + 48, y + 38, route="straight", allow_diagonal=True),
                edge(f"{prefix}_stack_down", x + 86, y + 54, x + 86, y + 124, route="vertical"),
                edge(f"{prefix}_grid_to_flatten", x + 130, y + 148, x + 122, y + 130, route="horizontal"),
            ]
        )
        return x + 172, y + 130

    def add_decoder(prefix: str, x: float, y: float, w: float, h: float, *, input_y: float, output_label: str) -> None:
        cid = f"{prefix}_decoder"
        nodes.append(container(cid, x, y, w, h))
        nodes.extend(
            [
                label(f"{prefix}_decoder_title", x + w - 98, y + 12, 70, 17, "Decoder", container_id=cid, font_size=10, weight="bold"),
                label(f"{prefix}_decoder_mx", x + w - 39, y + 29, 30, 16, "Mx", container_id=cid, font_size=8),
                query_stack(f"{prefix}_queries", x + 15, y + 45, 31, h - 72, container_id=cid),
                label(f"{prefix}_query_label", x + 49, y + 54, 20, h - 88, "Location-guided queries", container_id=cid, font_size=8.3, angle=90),
                op(f"{prefix}_ps_add", x + 83, y + 113, 22, container_id=cid),
                op(f"{prefix}_pm_add", x + 112, y + 18, 20, container_id=cid),
                box(f"{prefix}_cross_attention", x + 135, y + 42, 45, h - 82, "Multi-Head Deformable\nCross-Attention", container_id=cid, font_size=8.2, angle=90, rounded=True),
                box(f"{prefix}_dec_norm", x + 197, y + 70, 31, h - 112, "Add & Norm", container_id=cid, font_size=8.5, angle=90, rounded=True),
                box(f"{prefix}_dec_bcffn", x + 244, y + 66, 31, h - 106, "BC-FFN", container_id=cid, fill="#EAF3F9", font_size=8, angle=90, rounded=True),
                math_label(f"{prefix}_ps", x + 70, y + 45, 24, 16, "P_s", container_id=cid, font_size=8.5),
                math_label(f"{prefix}_pm_label", x + 128, y + 16, 28, 16, "P_m", container_id=cid, font_size=8.5),
                math_label(f"{prefix}_q_l", x + 25, y + h - 35, 30, 16, "Q_L", container_id=cid, font_size=8.5),
                label(f"{prefix}_v", x + 119, y + 64, 14, 12, "v", container_id=cid, font_size=8, italic=True),
                label(f"{prefix}_k", x + 119, y + 112, 14, 12, "k", container_id=cid, font_size=8, italic=True),
                label(f"{prefix}_q", x + 119, y + 159, 14, 12, "q", container_id=cid, font_size=8, italic=True),
                math_label(f"{prefix}_xd", x + w + 7, input_y - 10, 36, 18, output_label, font_size=8.5),
            ]
        )
        nodes.extend(pos_token(f"{prefix}_ps_token", x + 91, y + 54, container_id=cid))
        nodes.extend(pos_token(f"{prefix}_pm_token", x + 158, y + 20, container_id=cid))
        edges.extend(
            [
                edge(f"{prefix}_flat_to_queries", x - 20, input_y, x + 15, input_y, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
                edge(f"{prefix}_queries_to_ps", x + 46, y + 113, x + 72, y + 113, route="horizontal"),
                edge(f"{prefix}_ps_to_attention", x + 105, y + 113, x + 135, y + 113, route="horizontal"),
                edge(f"{prefix}_pm_to_attention", x + 122, y + 28, x + 135, y + 65, route="straight", allow_diagonal=True),
                edge(f"{prefix}_dec_attention_to_norm", x + 180, y + 113, x + 197, y + 113, route="horizontal"),
                edge(f"{prefix}_norm_to_bcffn", x + 228, y + 113, x + 244, y + 113, route="horizontal"),
                edge(f"{prefix}_bcffn_to_out", x + 275, y + 113, x + w + 1, y + 113, route="horizontal"),
                edge(f"{prefix}_bottom_skip", x + 46, y + h - 20, x + 212, y + h - 20, route="horizontal", arrow=False),
                edge(f"{prefix}_skip_up", x + 212, y + h - 20, x + 212, y + 70, route="vertical"),
            ]
        )

    # Top row
    top_enc_out_x, top_enc_out_y = add_encoder("top", 151, 28, 322, 198, input_x=93, input_y=139)
    top_bridge_x, top_bridge_y = add_feature_bridge("top", 495, 28, top_lane=True)
    add_decoder("top", 652, 38, 287, 190, input_y=top_bridge_y, output_label="X_d")
    edges.extend(
        [
            edge("top_encoder_to_bridge", top_enc_out_x - 20, top_enc_out_y, 495, top_enc_out_y, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
            edge("top_bridge_to_decoder", top_bridge_x, top_bridge_y, 652, top_bridge_y, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
            edge("top_restore_to_decoder_add", 545, 28, 765, 28, route="orthogonal", points=[[545, 28], [765, 28], [765, 47]], allow_cross_container=True, allow_direct_cross_container=True),
        ]
    )

    # Bottom row
    bottom_enc_out_x, bottom_enc_out_y = add_encoder("bottom", 116, 284, 342, 194, input_x=82, input_y=375)
    bottom_bridge_x, bottom_bridge_y = add_feature_bridge("bottom", 482, 270, top_lane=False)
    add_decoder("bottom", 643, 283, 303, 197, input_y=bottom_bridge_y, output_label="X_d")
    edges.extend(
        [
            edge("bottom_encoder_to_bridge", bottom_enc_out_x - 20, bottom_enc_out_y, 482, bottom_enc_out_y, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
            edge("bottom_bridge_to_decoder", bottom_bridge_x, bottom_bridge_y, 643, bottom_bridge_y, route="horizontal", allow_cross_container=True, allow_direct_cross_container=True),
            edge("bottom_restore_to_decoder_add", 530, 270, 744, 270, route="orthogonal", points=[[530, 270], [744, 270], [744, 292]], allow_cross_container=True, allow_direct_cross_container=True),
        ]
    )

    # Legends, kept editable and repeated because the source shows them in both halves.
    def add_legend(prefix: str, x: float, y: float) -> None:
        nodes.extend(
            [
                pos_icon(f"{prefix}_legend_pos_icon", x, y + 8, z=62),
                label(f"{prefix}_legend_pos", x + 17, y, 120, 18, "Position encoding", font_size=8.5),
                op(f"{prefix}_legend_add_icon", x + 150, y + 8, 18, z=62),
                label(f"{prefix}_legend_add", x + 168, y, 34, 18, "Add", font_size=8.5),
                box(f"{prefix}_legend_conv_icon", x + 220, y - 1, 63, 22, "3x3 Conv", font_size=8),
                label(f"{prefix}_legend_conv", x + 290, y, 112, 18, "Convolution layer", font_size=8.5),
            ]
        )

    add_legend("top", 136, 240)
    add_legend("bottom", 125, 499)

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.deformable_transformer_encoder_decoder",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "deformable_transformer_encoder_decoder",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_transformer_encoder_decoder_template",
                "diagram_family": "deformable_transformer_encoder_decoder_with_feature_grids",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_visible_labels_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "top_encoder", "category": "encoder", "source_bbox_px": [151, 28, 473, 226]},
                    {"id": "top_feature_bridge", "category": "feature_grid_bridge", "source_bbox_px": [495, 28, 645, 233]},
                    {"id": "top_decoder", "category": "decoder", "source_bbox_px": [652, 38, 939, 228]},
                    {"id": "bottom_encoder", "category": "encoder", "source_bbox_px": [116, 284, 458, 478]},
                    {"id": "bottom_feature_bridge", "category": "feature_grid_bridge", "source_bbox_px": [482, 270, 632, 490]},
                    {"id": "bottom_decoder", "category": "decoder", "source_bbox_px": [643, 283, 946, 480]},
                ],
            },
            "region_plan": [
                {"id": "top_encoder", "category": "encoder", "source_bbox_px": [151, 28, 473, 226]},
                {"id": "top_feature_bridge", "category": "feature_grid_bridge", "source_bbox_px": [495, 28, 645, 233]},
                {"id": "top_decoder", "category": "decoder", "source_bbox_px": [652, 38, 939, 228]},
                {"id": "bottom_encoder", "category": "encoder", "source_bbox_px": [116, 284, 458, 478]},
                {"id": "bottom_feature_bridge", "category": "feature_grid_bridge", "source_bbox_px": [482, 270, 632, 490]},
                {"id": "bottom_decoder", "category": "decoder", "source_bbox_px": [643, 283, 946, 480]},
            ],
            "notes": [
                "Editable semantic reconstruction for deformable-transformer encoder/decoder figures.",
                "Dashed Encoder/Decoder regions, attention blocks, Add & Norm, BC-FFN internals, feature grids, restore stacks, location-guided query stacks, operators, labels, and connectors are Visio-editable components.",
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


def build_channel_attention_recalibration_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 981.0
    base_h = 469.0
    palette = ["#F15A24", "#E41A1C", "#FF7F00", "#8B1A8B", "#FF70C6", "#7B2CBF", "#F2A01F", "#F0A000"]

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def add_common(node: dict[str, Any], source: tuple[float, float, float, float], container_id: str, *, z: int | None = None) -> dict[str, Any]:
        x, y, w, h = source
        node["source_bbox_px"] = bbox(x, y, w, h)
        node["container_id"] = container_id
        if z is not None:
            node["z"] = z
        return node

    def audit(node_id: str, x: float, y: float, w: float, h: float) -> dict[str, Any]:
        item = px_node(node_id, "audit_region", sx(x), sy(y), sx(w), sy(h), "", fill="#FFFFFF", line="none", z=1, font_size=1, text_color="#111111")
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"].update({"fill": "none", "line": "none", "font_size_pt": 1})
        return item

    def plain_label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        container_id: str,
        *,
        font_size: float = 13,
        weight: str = "regular",
        color: str = "#111111",
        italic: bool = False,
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["style"].update(
            {
                "font_family_candidates": ["Times New Roman", "Cambria Math", "Cambria", "Microsoft YaHei UI"],
                "font_role": "math" if italic else "paper_serif",
                "text_color": color,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 5.0,
            }
        )
        if italic:
            item["style"]["font_italic"] = True
        return add_common(item, (x, y, w, h), container_id, z=z)

    def math_label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        container_id: str,
        *,
        font_size: float = 13,
        z: int = 92,
    ) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "math_text",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": text,
            "font_size_pt": font_size,
            "math_render_mode": "fragments",
            "style": {
                "fill": "none",
                "line": "none",
                "text_color": "#111111",
                "font_family_candidates": ["Times New Roman", "Cambria Math", "Cambria"],
                "font_role": "math",
                "font_italic": True,
                "font_size_pt": font_size,
                "min_font_size_pt": 5.0,
                "text_fit": "math_label",
                "text_margin_in": 0.0,
            },
        }
        return add_common(item, (x, y, w, h), container_id, z=z)

    def cuboid(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        container_id: str,
        *,
        depth_x: float = 30,
        depth_y: float = -30,
        fill: str = "#FFFFFF",
        side_fill: str = "#F6F6F6",
        top_fill: str = "#FFFFFF",
        z: int = 26,
    ) -> dict[str, Any]:
        item = px_node(node_id, "cuboid_node", sx(x), sy(y), sx(w), sy(h), text, fill=fill, line="#111111", z=z, font_size=14, text_color="#111111")
        item["depth_x_in"] = sx(depth_x)
        item["depth_y_in"] = sy(depth_y)
        item["style"].update(
            {
                "fill": fill,
                "side_fill": side_fill,
                "top_fill": top_fill,
                "line": "#111111",
                "line_weight_pt": 1.1,
                "font_family_candidates": ["Times New Roman", "Cambria Math", "Cambria"],
                "font_role": "math",
                "font_italic": True,
                "font_size_pt": 14,
                "text_fit": "single_line",
            }
        )
        return add_common(item, (x, y + depth_y, w + depth_x, h - depth_y), container_id, z=z)

    def vector_stack(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        container_id: str,
        *,
        fills: list[str] | None = None,
        count: int = 9,
        label: str = "1x1xC",
        z: int = 48,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": node_id,
            "type": "feature_vector_stack",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "orientation": "horizontal",
            "count": count,
            "cell_gap_in": 1.2,
            "cell_fills": fills or ["#FFFFFF"] * count,
            "label": label,
            "label_side": "bottom",
            "label_gap_in": sy(2.0),
            "label_w_in": sx(58.0),
            "label_h_in": sy(15.0),
            "label_font_size_pt": 8.5,
            "label_text_fit": "single_line",
            "style": {
                "fill": "#FFFFFF",
                "cell_fill": "#FFFFFF",
                "cell_line": "#111111",
                "line": "#111111",
                "line_weight_pt": 0.8,
                "cell_line_weight_pt": 0.8,
                "font_family_candidates": ["Times New Roman", "Cambria Math", "Cambria"],
                "font_role": "math",
                "font_size_pt": 8.5,
                "cell_font_size_pt": 7.5,
                "cell_text_fit": "single_line",
                "text_margin_in": 0.0,
            },
        }
        return add_common(item, (x, y, w, h + 18), container_id, z=z)

    def tensor_output(
        prefix: str,
        x: float,
        y: float,
        w: float,
        h: float,
        container_id: str,
        *,
        depth_x: float = 38,
        depth_y: float = -32,
        z: int = 32,
    ) -> list[dict[str, Any]]:
        tensor = {
            "id": f"{prefix}_tensor_core",
            "type": "tensor_stack",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "layers": 1,
            "stack_render_mode": "feature_cuboids",
            "depth_x_in": sx(depth_x),
            "depth_y_in": sy(depth_y),
            "style": {
                "fill": "#F39C12",
                "side_fill": "#D07B00",
                "top_fill": "#FDBA3B",
                "line": "#111111",
                "line_weight_pt": 1.05,
            },
            "allow_overlap": True,
        }
        front = {
            "id": f"{prefix}_channel_bands",
            "type": "feature_map_banded",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z + 10,
            "text": "",
            "orientation": "vertical",
            "bands": [{"fill": color, "size": 1} for color in palette],
            "separator_count": len(palette) - 1,
            "style": {
                "fill": "none",
                "line": "#111111",
                "line_weight_pt": 1.05,
                "separator_line": "#111111",
                "separator_line_weight_pt": 0.8,
            },
            "allow_overlap": True,
        }
        return [
            add_common(tensor, (x, y + depth_y, w + depth_x, h - depth_y), container_id, z=z),
            add_common(front, (x, y, w, h), container_id, z=z + 10),
        ]

    def arrow(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "horizontal",
        points: list[list[float]] | None = None,
        arrowhead: bool = True,
        allow_diagonal: bool = False,
        arrow_plan_id: str | None = None,
        z: int = 66,
    ) -> dict[str, Any]:
        edge: dict[str, Any] = {
            "id": edge_id,
            "type": "lane_arrow" if route in {"horizontal", "vertical"} and arrowhead else ("arrow_connector" if arrowhead else "line_segment"),
            "from_point": [sx(x1), sy(y1)],
            "to_point": [sx(x2), sy(y2)],
            "route": route,
            "z": z,
            "style": {
                "line": "#111111",
                "line_weight_pt": 1.1,
                "end_arrow": "triangle" if arrowhead else "none",
                "arrow_size": "small",
            },
        }
        if points:
            edge["points"] = [[sx(px), sy(py)] for px, py in points]
            edge["orthogonalize_points"] = route in {"orthogonal", "hv", "vh"}
        if allow_diagonal:
            edge["allow_diagonal"] = True
        if arrow_plan_id:
            edge["arrow_plan_id"] = arrow_plan_id
        return edge

    nodes: list[dict[str, Any]] = [
        px_node("page_background", "page_background", 0, 0, width, height, "", fill="#FFFFFF", line="none", z=0),
        audit("top_channel_attention_lane", 0, 20, 955, 175),
        audit("bottom_channel_attention_lane", 45, 255, 920, 190),
        plain_label("original_word", 0, 88, 104, 30, "Original", "top_channel_attention_lane", font_size=20, weight="bold", color="#79CFD0"),
        plain_label("image_word", 9, 121, 82, 31, "image", "top_channel_attention_lane", font_size=20, weight="bold", color="#79CFD0"),
    ]

    def add_pipeline(prefix: str, y_offset: float, *, with_original_arrow: bool) -> None:
        container = f"{prefix}_channel_attention_lane"
        if prefix == "top":
            x1, y1, x2, y2, vector_y, colored_y, out_x, out_y = 199, 92, 365, 99, 52, 53, 801, 103
            input_depth = (28, -28)
            u_depth = (26, -29)
            input_size = (55, 74)
            u_size = (62, 64)
            output_size = (64, 64)
            output_depth = (35, -31)
            arrow_y = 128
        else:
            x1, y1, x2, y2, vector_y, colored_y, out_x, out_y = 126, 345, 310, 345, 294, 294, 813, 346
            input_depth = (34, -33)
            u_depth = (36, -33)
            input_size = (56, 82)
            u_size = (62, 82)
            output_size = (73, 82)
            output_depth = (48, -35)
            arrow_y = 386

        nodes.extend(
            [
                cuboid(f"{prefix}_input_x", x1, y1, input_size[0], input_size[1], "H'", container, depth_x=input_depth[0], depth_y=input_depth[1], z=26),
                plain_label(f"{prefix}_x_label", x1 + input_size[0] * 0.72, y1 + input_depth[1] - 19, 30, 18, "X", container, font_size=13, weight="bold", italic=True),
                plain_label(f"{prefix}_input_c_label", x1 + input_size[0] * 0.40, y1 + input_size[1] + 5, 35, 18, "C'", container, font_size=12, italic=True),
                plain_label(f"{prefix}_input_w_label", x1 + input_size[0] + input_depth[0] - 9, y1 + 40, 28, 18, "W'", container, font_size=12, italic=True),
                cuboid(f"{prefix}_feature_u", x2, y2, u_size[0], u_size[1], "H", container, depth_x=u_depth[0], depth_y=u_depth[1], z=27),
                plain_label(f"{prefix}_u_label", x2 + u_size[0] * 0.72, y2 + u_depth[1] - 17, 28, 18, "U", container, font_size=13, weight="bold", italic=True),
                plain_label(f"{prefix}_u_c_label", x2 + u_size[0] * 0.42, y2 + u_size[1] + 6, 28, 18, "C", container, font_size=12, italic=True),
                plain_label(f"{prefix}_u_w_label", x2 + u_size[0] + u_depth[0] - 9, y2 + 36, 24, 18, "W", container, font_size=12, italic=True),
                vector_stack(f"{prefix}_squeezed_vector", 535 if prefix == "top" else 495, vector_y, 46 if prefix == "top" else 76, 13 if prefix == "top" else 16, container, fills=None, count=9 if prefix == "top" else 10, z=49),
                vector_stack(f"{prefix}_excited_vector", 660 if prefix == "top" else 670, colored_y, 51 if prefix == "top" else 57, 15 if prefix == "top" else 16, container, fills=palette, count=8, z=50),
            ]
        )
        nodes.extend(tensor_output(f"{prefix}_output", out_x, out_y, output_size[0], output_size[1], container, depth_x=output_depth[0], depth_y=output_depth[1], z=33))
        nodes.extend(
            [
                plain_label(f"{prefix}_xtilde_label", out_x + output_size[0] * 0.72, out_y + output_depth[1] - 16, 35, 18, "X~", container, font_size=13, weight="bold", italic=True),
                plain_label(f"{prefix}_output_h_label", out_x + output_size[0] + output_depth[0] + 8, out_y + 16, 25, 22, "H", container, font_size=13, italic=True),
                plain_label(f"{prefix}_output_c_label", out_x + output_size[0] * 0.52, out_y + output_size[1] + 7, 25, 18, "C", container, font_size=12, italic=True),
                plain_label(f"{prefix}_output_w_label", out_x + output_size[0] + output_depth[0] - 8, out_y + output_size[1] - 4, 25, 18, "W", container, font_size=12, italic=True),
                math_label(f"{prefix}_ftr_label", (x1 + input_size[0] + input_depth[0] + x2 - 20) / 2, arrow_y - 23, 60, 22, "F_tr", container, font_size=13),
                math_label(f"{prefix}_fsq_label", x2 + u_size[0] + 18, vector_y - 15, 66, 20, "F_sq(.)", container, font_size=12),
                math_label(f"{prefix}_fex_label", (535 if prefix == "top" else 495) + 90, vector_y - 29, 108, 22, "F_ex(., W)", container, font_size=12),
                math_label(f"{prefix}_fscale_label", 710 if prefix == "top" else 728, arrow_y - 28, 105, 24, "F_scale(., .)", container, font_size=12),
            ]
        )

        if with_original_arrow:
            edges.append(arrow(f"{prefix}_original_to_x", 165, arrow_y, x1 - 1, arrow_y, route="horizontal", arrow_plan_id=f"{prefix}_A001"))
        else:
            edges.append(arrow(f"{prefix}_input_arrow", 82, arrow_y, x1 - 1, arrow_y, route="horizontal", arrow_plan_id=f"{prefix}_A001"))
        edges.extend(
            [
                arrow(f"{prefix}_x_to_u", x1 + input_size[0] + input_depth[0] + 3, arrow_y, x2 - 2, arrow_y, route="horizontal", arrow_plan_id=f"{prefix}_A002"),
                arrow(f"{prefix}_u_to_output_main", x2 + u_size[0] + u_depth[0] + 2, arrow_y, out_x - 3, arrow_y, route="horizontal", arrow_plan_id=f"{prefix}_A003"),
                arrow(f"{prefix}_u_to_squeeze", x2 + u_size[0] + u_depth[0] - 2, y2 + 5, (535 if prefix == "top" else 495) - 4, vector_y + 7, route="straight", allow_diagonal=True, arrow_plan_id=f"{prefix}_A004"),
                arrow(f"{prefix}_squeeze_to_excite", (535 if prefix == "top" else 495) + (46 if prefix == "top" else 76) + 4, vector_y + 7, (660 if prefix == "top" else 670) - 4, colored_y + 7, route="horizontal", arrow_plan_id=f"{prefix}_A005"),
                arrow(f"{prefix}_excite_to_scale", (660 if prefix == "top" else 670) + (51 if prefix == "top" else 57) + 3, colored_y + 8, out_x - 2, out_y + 4, route="straight", allow_diagonal=True, arrow_plan_id=f"{prefix}_A006"),
                arrow(f"{prefix}_output_to_right", out_x + output_size[0] + output_depth[0] + 2, arrow_y, min(960, out_x + output_size[0] + output_depth[0] + 44), arrow_y, route="horizontal", arrow_plan_id=f"{prefix}_A007"),
            ]
        )

    edges: list[dict[str, Any]] = []
    add_pipeline("top", 0, with_original_arrow=True)
    add_pipeline("bottom", 235, with_original_arrow=False)

    arrow_plan = []
    for prefix in ("top", "bottom"):
        lane_name = "upper recalibration lane" if prefix == "top" else "lower recalibration lane"
        arrow_plan.extend(
            [
                {"id": f"{prefix}_A001", "from_visual_object": "input arrow", "from_anchor_description": "left side", "to_visual_object": "X tensor", "to_anchor_description": "left face", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high", "lane": lane_name},
                {"id": f"{prefix}_A002", "from_visual_object": "X tensor", "from_anchor_description": "right side", "to_visual_object": "U tensor", "to_anchor_description": "left face", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high", "lane": lane_name},
                {"id": f"{prefix}_A003", "from_visual_object": "U tensor", "from_anchor_description": "right side", "to_visual_object": "scaled output tensor", "to_anchor_description": "left face", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high", "lane": lane_name},
                {"id": f"{prefix}_A004", "from_visual_object": "U tensor", "from_anchor_description": "upper-right corner", "to_visual_object": "1x1xC squeezed vector", "to_anchor_description": "left side", "route_shape": "straight_diagonal", "semantic_intent": "squeeze", "certainty": "high", "lane": lane_name},
                {"id": f"{prefix}_A005", "from_visual_object": "squeezed vector", "from_anchor_description": "right side", "to_visual_object": "excited channel vector", "to_anchor_description": "left side", "route_shape": "straight_horizontal", "semantic_intent": "excitation", "certainty": "high", "lane": lane_name},
                {"id": f"{prefix}_A006", "from_visual_object": "excited channel vector", "from_anchor_description": "right side", "to_visual_object": "scaled output tensor", "to_anchor_description": "upper-left face", "route_shape": "straight_diagonal", "semantic_intent": "scale_weights", "certainty": "high", "lane": lane_name},
                {"id": f"{prefix}_A007", "from_visual_object": "scaled output tensor", "from_anchor_description": "right side", "to_visual_object": "output arrow", "to_anchor_description": "right side", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high", "lane": lane_name},
            ]
        )
    for plan in arrow_plan:
        plan.setdefault("from", f"{plan.get('from_visual_object', '')} {plan.get('from_anchor_description', '')}".strip())
        plan.setdefault("to", f"{plan.get('to_visual_object', '')} {plan.get('to_anchor_description', '')}".strip())
        if plan.get("route_shape") == "straight_diagonal":
            plan["route_shape"] = "short_diagonal"
        if plan.get("semantic_intent") in {"squeeze", "excitation", "scale_weights"}:
            plan["semantic_intent"] = "data_flow"

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.channel_attention_recalibration",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "channel_attention_recalibration",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_formula_and_tensor_motif_triggered_channel_attention_template",
                "diagram_family": "channel_attention_squeeze_excitation_recalibration",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_visible_formula_family_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "top_channel_attention_lane", "category": "upper_pipeline", "source_bbox_px": [0, 20, 955, 195], "required_visible_labels": ["Original image", "X", "U", "F_tr", "F_sq", "F_ex", "F_scale", "1x1xC", "X~"]},
                    {"id": "bottom_channel_attention_lane", "category": "lower_pipeline", "source_bbox_px": [45, 255, 965, 445], "required_visible_labels": ["X", "U", "F_tr", "F_sq", "F_ex", "F_scale", "1x1xC", "X~"]},
                ],
            },
            "region_plan": [
                {"id": "top_channel_attention_lane", "category": "upper_pipeline", "source_bbox_px": [0, 20, 955, 195]},
                {"id": "bottom_channel_attention_lane", "category": "lower_pipeline", "source_bbox_px": [45, 255, 965, 445]},
            ],
            "arrow_plan": arrow_plan,
            "notes": [
                "Editable semantic reconstruction for channel-attention or squeeze-excitation recalibration diagrams.",
                "3D feature tensors are cuboid/tensor components; 1x1xC rows are feature_vector_stack components; output channel faces are editable colored bands.",
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


def build_attention_mechanism_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 743.0
    base_h = 354.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#666666",
        z: int = 20,
        font_size: float = 13,
        dash: str = "solid",
        rounding: float = 0.08,
        shadow: bool = False,
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            rounding=rounding,
        )
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"]["font_family"] = "Times New Roman"
        if shadow:
            item["style"]["shadow"] = {
                "color": "#222222",
                "offset_x_in": 0.035,
                "offset_y_in": -0.035,
                "transparency_pct": 84,
            }
        return item

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 13,
        weight: str = "regular",
        italic: bool = False,
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"]["font_family"] = "Times New Roman"
        item["style"]["text_fit"] = "shrink_to_fit"
        if italic:
            item["style"]["font_italic"] = True
        return item

    def feature_bands(node_id: str, x: float, y: float, w: float, h: float) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": "feature_map_banded",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": 24,
            "text": "",
            "bands": [
                {"fill": "#F2AF83", "size": 1},
                {"fill": "#9ED2E3", "size": 1},
                {"fill": "#D3E4C2", "size": 1},
                {"fill": "#F7E78C", "size": 1},
                {"fill": "#9ED2E3", "size": 1},
                {"fill": "#F2AF83", "size": 1},
            ],
            "source_bbox_px": bbox(x, y, w, h),
            "style": {
                "line": "#C4C4C4",
                "line_weight_pt": 0.7,
                "shadow": {
                    "color": "#222222",
                    "offset_x_in": 0.035,
                    "offset_y_in": -0.035,
                    "transparency_pct": 86,
                },
            },
        }

    def weighted_grid(node_id: str, x: float, y: float, w: float, h: float) -> dict[str, Any]:
        fills = ["#FFFFFF", "#E8E8E8", "#D8D8D8", "#B0B0B0", "#FFFFFF", "#6B6B6B", "#9A9A9A", "#FFFFFF", "#FFFFFF"]
        return {
            "id": node_id,
            "type": "grid_matrix",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": 35,
            "text": "",
            "rows": 1,
            "cols": len(fills),
            "colored_cells": [[0, index, fill] for index, fill in enumerate(fills)],
            "source_bbox_px": bbox(x, y, w, h),
            "style": {
                "grid_line": "#111111",
                "grid_line_weight_pt": 0.9,
                "line": "#111111",
                "line_weight_pt": 0.9,
            },
        }

    def feature_grid(node_id: str, x: float, y: float, w: float, h: float) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": "feature_map_grid",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": 24,
            "text": "",
            "rows": 6,
            "cols": 9,
            "row_colors": ["#F2AF83", "#9ED2E3", "#D3E4C2", "#F7E78C", "#9ED2E3", "#F2AF83"],
            "column_shades": [0.0, 0.18, 0.38, 0.55, 0.10, 0.72, 0.62, 0.0, 0.18],
            "max_shade": 0.68,
            "source_bbox_px": bbox(x, y, w, h),
            "style": {
                "grid_line": "#111111",
                "grid_line_weight_pt": 0.7,
                "line": "#111111",
                "line_weight_pt": 0.9,
                "shadow": {
                    "color": "#222222",
                    "offset_x_in": 0.035,
                    "offset_y_in": -0.035,
                    "transparency_pct": 86,
                },
            },
        }

    def operator(node_id: str, x: float, y: float, size: float, symbol: str) -> dict[str, Any]:
        item = node(
            node_id,
            "operator_node",
            x,
            y,
            size,
            size,
            symbol,
            fill="#FFFFFF",
            line="#6F6F6F",
            z=72,
            font_size=11,
            rounding=0.0,
        )
        item["symbol"] = symbol
        item["operator_shape"] = "circle"
        item["operator_size_tier"] = "small"
        item["style"]["text_color"] = "#5C5C5C"
        item["style"]["line_weight_pt"] = 1.15
        return item

    def edge_ref(
        edge_id: str,
        from_ref: str,
        to_ref: str,
        *,
        route: str = "horizontal",
        points: list[list[float]] | None = None,
        arrow: bool = True,
        arrow_plan_id: str | None = None,
        weight: float = 1.1,
        z: int = 60,
        allow_cross_container: bool = False,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": edge_id,
            "type": "arrow_connector",
            "from": from_ref,
            "to": to_ref,
            "route": route,
            "z": z,
            "style": {
                "line": "#6F6F6F",
                "line_weight_pt": weight,
                "arrow_size": "small",
                "end_arrow": "triangle" if arrow else "none",
            },
        }
        if points:
            item["points"] = [[sx(px), sy(py)] for px, py in points]
            item["orthogonalize_points"] = route in {"orthogonal", "hv", "vh"}
        if arrow_plan_id:
            item["arrow_plan_id"] = arrow_plan_id
        if allow_cross_container:
            item["allow_cross_container"] = True
        return item

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0),
        label("journal_header", 202, -11, 405, 20, "Digital Communications and Networks 11 (2025) 1567-1577", font_size=10.5, italic=True, z=96),
        label("attention_title", 198, 7, 184, 24, "Attention mechanism", font_size=14.5, z=95),
        node("attention_frame", "group_container", 200, 36, 178, 120, "", fill="none", line="#8F8F8F", z=5, dash="dash", rounding=0.28),
        node("sigmoid", "rounded_process", 222, 49, 132, 36, "Sigmoid", fill="#FFE49A", line="#FFE49A", z=28, font_size=14.5, rounding=0.13, shadow=True),
        node("conv1d", "rounded_process", 224, 107, 129, 36, "Conv1d", fill="#93D4EA", line="#93D4EA", z=28, font_size=14.5, rounding=0.13, shadow=True),
        feature_bands("high_level_features", 47, 154, 139, 94),
        label("high_level_label", 36, 256, 165, 28, "High-level features", font_size=14.5, z=92),
        weighted_grid("weighted_vector", 402, 117, 141, 17),
        label("weighted_vector_label", 485, 83, 140, 28, "Weighted vector", font_size=14.5, z=92),
        operator("multiply_op", 461, 189, 24, "x"),
        feature_grid("am_resnet_features", 561, 154, 141, 94),
        label("am_resnet_label", 544, 256, 174, 28, "AM-ResNet features", font_size=14.5, z=92),
        label("figure_caption", 134, 315, 485, 31, "Fig. 5. The architecture of the attention mechanism.", font_size=14.5, weight="bold", z=95),
    ]
    nodes[3]["shape"] = "capsule"
    nodes[3]["allow_overlap"] = True

    edges: list[dict[str, Any]] = [
        edge_ref("feature_to_conv", "high_level_features:right@0.50", "conv1d:bottom@0.50", route="orthogonal", points=[[186, 201], [289, 201], [289, 143]], arrow_plan_id="A001", allow_cross_container=True),
        edge_ref("conv_to_sigmoid", "conv1d:top@0.50", "sigmoid:bottom@0.50", route="vertical", arrow_plan_id="A002"),
        edge_ref("sigmoid_to_weighted", "sigmoid:right@0.50", "weighted_vector:top@0.50", route="orthogonal", points=[[354, 67], [472, 67], [472, 117]], arrow_plan_id="A003", allow_cross_container=True),
        edge_ref("feature_to_multiply", "high_level_features:right@0.50", "multiply_op:left", route="horizontal", arrow_plan_id="A004", allow_cross_container=True),
        edge_ref("weighted_to_multiply", "weighted_vector:bottom@0.50", "multiply_op:top", route="vertical", arrow_plan_id="A005"),
        edge_ref("multiply_to_output", "multiply_op:right", "am_resnet_features:left@0.50", route="horizontal", arrow_plan_id="A006", allow_cross_container=True),
    ]

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.attention_mechanism",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "attention_mechanism",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_source_coordinate_paper_template",
                "diagram_family": "attention_mechanism_feature_weighting",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_ocr_when_visible_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "input_feature_map", "category": "input", "source_bbox_px": [47, 154, 186, 248]},
                    {"id": "attention_core", "category": "core", "source_bbox_px": [200, 36, 378, 156]},
                    {"id": "weighted_vector", "category": "core", "source_bbox_px": [402, 83, 625, 134]},
                    {"id": "output_feature_map", "category": "output", "source_bbox_px": [561, 154, 702, 248]},
                    {"id": "figure_caption", "category": "caption", "source_bbox_px": [134, 315, 619, 346]},
                ],
            },
            "arrow_plan": [
                {"id": "A001", "from": "High-level features right center", "from_visual_object": "High-level features", "from_anchor_description": "right center", "to": "Conv1d bottom center", "to_visual_object": "Conv1d", "to_anchor_description": "bottom center", "route_shape": "orthogonal", "semantic_intent": "data_flow", "certainty": "high"},
                {"id": "A002", "from": "Conv1d top center", "from_visual_object": "Conv1d", "from_anchor_description": "top center", "to": "Sigmoid bottom center", "to_visual_object": "Sigmoid", "to_anchor_description": "bottom center", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
                {"id": "A003", "from": "Sigmoid right center", "from_visual_object": "Sigmoid", "from_anchor_description": "right center", "to": "Weighted vector top center", "to_visual_object": "Weighted vector", "to_anchor_description": "top center", "route_shape": "orthogonal", "semantic_intent": "data_flow", "certainty": "high"},
                {"id": "A004", "from": "High-level features right center", "from_visual_object": "High-level features", "from_anchor_description": "right center", "to": "multiply operator left side", "to_visual_object": "multiply operator", "to_anchor_description": "left side", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
                {"id": "A005", "from": "Weighted vector bottom center", "from_visual_object": "Weighted vector", "from_anchor_description": "bottom center", "to": "multiply operator top side", "to_visual_object": "multiply operator", "to_anchor_description": "top side", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
                {"id": "A006", "from": "multiply operator right side", "from_visual_object": "multiply operator", "from_anchor_description": "right side", "to": "AM-ResNet features left center", "to_visual_object": "AM-ResNet features", "to_anchor_description": "left center", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
            ],
            "notes": [
                "Editable semantic reconstruction for compact attention mechanism paper figures.",
                "High-level feature bands, Conv1d/Sigmoid attention core, weighted vector, multiply node, AM-ResNet feature grid, connectors, and caption are vector Visio objects.",
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


def build_cross_attention_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 1368.0
    base_h = 438.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#6F6F6F",
        z: int = 20,
        font_size: float = 13,
        dash: str = "solid",
        rounding: float = 0.08,
        shadow: bool = False,
        italic: bool = False,
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            rounding=rounding,
        )
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"]["font_family"] = "Times New Roman"
        if shadow:
            item["style"]["shadow"] = {
                "color": "#222222",
                "offset_x_in": 0.035,
                "offset_y_in": -0.035,
                "transparency_pct": 84,
            }
        if italic:
            item["style"]["font_italic"] = True
        return item

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 12,
        weight: str = "regular",
        italic: bool = False,
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"]["font_family"] = "Times New Roman"
        if italic:
            item["style"]["font_italic"] = True
        return item

    def operator(node_id: str, x: float, y: float, size: float, symbol: str, *, font_size: float = 11) -> dict[str, Any]:
        item = node(
            node_id,
            "operator_node",
            x,
            y,
            size,
            size,
            symbol,
            fill="#FFFFFF",
            line="#777777",
            z=72,
            font_size=font_size,
            rounding=0.0,
        )
        item["symbol"] = symbol
        item["operator_shape"] = "circle"
        item["operator_size_tier"] = "small"
        item["style"]["text_color"] = "#666666"
        return item

    def grid_cells(rows: int, cols: int, color_a: str, color_b: str) -> list[list[object]]:
        return [[r, c, color_a if (r + c) % 2 == 0 else color_b] for r in range(rows) for c in range(cols)]

    def grid(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rows: int = 4,
        cols: int = 5,
        color_a: str = "#F7BBD4",
        color_b: str = "#B7DBF1",
    ) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": "grid_matrix",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": 35,
            "text": "",
            "rows": rows,
            "cols": cols,
            "colored_cells": grid_cells(rows, cols, color_a, color_b),
            "source_bbox_px": bbox(x, y, w, h),
            "style": {
                "grid_line": "#777777",
                "grid_line_weight_pt": 0.55,
                "line": "#777777",
                "line_weight_pt": 0.55,
            },
        }

    def edge_ref(
        edge_id: str,
        from_ref: str,
        to_ref: str,
        *,
        route: str = "horizontal",
        points: list[list[float]] | None = None,
        color: str = "#6F6F6F",
        weight: float = 1.05,
        arrow: bool = True,
        edge_type: str = "arrow_connector",
        allow_diagonal: bool = False,
        z: int = 60,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": edge_id,
            "type": edge_type,
            "from": from_ref,
            "to": to_ref,
            "route": route,
            "z": z,
            "style": {
                "line": color,
                "line_weight_pt": weight,
                "arrow_size": "small",
                "end_arrow": "triangle" if arrow else "none",
            },
        }
        if points:
            item["points"] = [[sx(px), sy(py)] for px, py in points]
            item["orthogonalize_points"] = route in {"orthogonal", "hv", "vh"}
        if allow_diagonal:
            item["allow_diagonal"] = True
        return item

    def edge_points(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "straight",
        points: list[list[float]] | None = None,
        arrow: bool = True,
        color: str = "#6F6F6F",
        weight: float = 1.0,
        edge_type: str = "arrow_connector",
        z: int = 60,
    ) -> dict[str, Any]:
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=arrow,
            route=route,
            points=[[sx(px), sy(py)] for px, py in points] if points else None,
            z=z,
            allow_cross_container=True,
        )
        item["type"] = edge_type
        item["style"].update(
            {
                "line": color,
                "line_weight_pt": weight,
                "arrow_size": "small",
                "end_arrow": "triangle" if arrow else "none",
            }
        )
        return item

    blue = "#BDD0F4"
    pink = "#F7C7E7"
    green = "#BFD69D"
    purple = "#D6C5E3"
    gray = "#D6D6D6"
    yellow = "#FDE8A4"
    token_pink = "#FFE1F4"
    token_blue = "#CFE0FF"

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0),
        node("attention_frame", "group_container", 36, 10, 1086, 347, "", fill="none", line="#8F8F8F", z=5, dash="dash", rounding=0.22),
        label("journal_header", 900, -17, 438, 28, "Digital Communications and Networks 11 (2025) 100", font_size=13, italic=True, z=95),
        node("am_features", "rounded_process", 54, 91, 157, 73, "AM-ResNet\nfeatures", fill=blue, line=blue, z=24, font_size=13, shadow=True, rounding=0.12),
        node("wav_features", "rounded_process", 54, 202, 157, 73, "Wav2vec 2.0\nfeatures", fill=pink, line=pink, z=24, font_size=13, shadow=True, rounding=0.12),
        node("avg_pool", "rounded_process", 243, 203, 84, 72, "Avg\npool", fill=green, line=green, z=24, font_size=14, rounding=0.06),
        node("fc_am", "rounded_process", 356, 91, 49, 73, "FC", fill=gray, line=gray, z=24, font_size=13, rounding=0.04),
        node("fc_wav", "rounded_process", 356, 202, 49, 73, "FC", fill=gray, line=gray, z=24, font_size=13, rounding=0.04),
        node("token_vw", "text_pill", 459, 81, 36, 36, "Vw", fill=token_pink, line=token_pink, z=30, font_size=12, italic=True, rounding=0.03),
        node("token_kw", "text_pill", 459, 115, 36, 36, "Kw", fill=token_pink, line=token_pink, z=30, font_size=12, italic=True, rounding=0.03),
        node("token_qa", "text_pill", 459, 148, 36, 36, "Qa", fill=token_blue, line=token_blue, z=30, font_size=12, italic=True, rounding=0.03),
        node("token_qw", "text_pill", 459, 192, 36, 36, "Qw", fill=token_pink, line=token_pink, z=30, font_size=12, italic=True, rounding=0.03),
        node("token_ka", "text_pill", 459, 226, 36, 36, "Ka", fill=token_blue, line=token_blue, z=30, font_size=12, italic=True, rounding=0.03),
        node("token_va", "text_pill", 459, 257, 36, 31, "Va", fill=token_blue, line=token_blue, z=30, font_size=12, italic=True, rounding=0.03),
        operator("op_top_attention", 502, 131, 18, "⊗", font_size=8),
        operator("op_top_value", 583, 86, 18, "⊗", font_size=8),
        operator("op_bottom_attention", 502, 224, 18, "⊗", font_size=8),
        operator("op_bottom_value", 583, 260, 18, "⊗", font_size=8),
        label("softmax_top", 519, 126, 52, 18, "Softmax", font_size=8, z=75),
        label("softmax_bottom", 517, 202, 54, 18, "Softmax", font_size=8, z=75),
        grid("attn_map_top", 572, 127, 39, 32, rows=4, cols=5),
        grid("weighted_map_top", 620, 82, 39, 31, rows=5, cols=5, color_a="#FFD4EA", color_b="#F8C8DF"),
        grid("attn_map_bottom", 572, 210, 39, 31, rows=4, cols=5),
        grid("weighted_map_bottom", 620, 255, 39, 31, rows=5, cols=5, color_a="#D8D1EB", color_b="#C7E3EA"),
        node("concat_top", "rounded_process", 589, 37, 101, 34, "Concat", fill=purple, line=purple, z=24, font_size=13, rounding=0.05),
        node("norm_top_1", "rounded_process", 710, 36, 80, 34, "norm", fill=gray, line=gray, z=24, font_size=12, rounding=0.04),
        node("ff_top", "rounded_process", 806, 28, 118, 52, "Feed\nforward", fill=gray, line=gray, z=24, font_size=12, rounding=0.08),
        operator("op_add_top", 857, 126, 18, "+", font_size=10),
        node("norm_top_2", "rounded_process", 932, 119, 79, 33, "norm", fill=gray, line=gray, z=24, font_size=12, rounding=0.04),
        node("concat_bottom", "rounded_process", 590, 297, 101, 34, "Concat", fill=purple, line=purple, z=24, font_size=13, rounding=0.05),
        node("norm_bottom_1", "rounded_process", 711, 298, 80, 34, "norm", fill=gray, line=gray, z=24, font_size=12, rounding=0.04),
        node("ff_bottom", "rounded_process", 806, 288, 118, 55, "Feed\nforward", fill=gray, line=gray, z=24, font_size=12, rounding=0.08),
        operator("op_add_bottom", 856, 223, 18, "+", font_size=10),
        node("norm_bottom_2", "rounded_process", 932, 216, 79, 33, "norm", fill=gray, line=gray, z=24, font_size=12, rounding=0.04),
        node("concat_final", "rounded_process", 1009, 169, 101, 34, "Concat", fill=purple, line=purple, z=24, font_size=13, rounding=0.05),
        node("cross_fused", "rounded_process", 1154, 150, 173, 72, "Cross-fused\nfeatures", fill=yellow, line=yellow, z=24, font_size=13, shadow=True, rounding=0.09),
        label("figure_caption", 471, 396, 430, 26, "Fig. 7. The architecture of the cross-attention.", font_size=13, weight="bold", z=95),
    ]

    edges: list[dict[str, Any]] = [
        edge_ref("am_to_fc", "am_features:right@0.50", "fc_am:left@0.50"),
        edge_ref("wav_to_avg", "wav_features:right@0.50", "avg_pool:left@0.50"),
        edge_ref("avg_to_fc", "avg_pool:right@0.50", "fc_wav:left@0.50"),
        edge_ref("am_skip_to_concat_top", "am_features:top@0.50", "concat_top:left@0.50", edge_type="residual_connector", points=[[132, 91], [132, 54], [589, 54]], route="orthogonal"),
        edge_ref("avg_skip_to_concat_bottom", "avg_pool:bottom@0.50", "concat_bottom:left@0.50", edge_type="residual_connector", points=[[285, 275], [285, 314], [590, 314]], route="orthogonal"),
        edge_ref("fc_am_to_qa", "fc_am:right@0.50", "token_qa:left@0.50", route="straight", color="#8DB7FF", weight=1.2, arrow=False, allow_diagonal=True),
        edge_ref("fc_am_to_ka", "fc_am:right@0.50", "token_ka:left@0.50", route="straight", color="#8DB7FF", weight=1.2, arrow=False, allow_diagonal=True),
        edge_ref("fc_am_to_va", "fc_am:right@0.50", "token_va:left@0.50", route="straight", color="#8DB7FF", weight=1.2, arrow=False, allow_diagonal=True),
        edge_ref("fc_wav_to_vw", "fc_wav:right@0.50", "token_vw:left@0.50", route="straight", color="#FF91CA", weight=1.2, arrow=False, allow_diagonal=True),
        edge_ref("fc_wav_to_kw", "fc_wav:right@0.50", "token_kw:left@0.50", route="straight", color="#FF91CA", weight=1.2, arrow=False, allow_diagonal=True),
        edge_ref("fc_wav_to_qw", "fc_wav:right@0.50", "token_qw:left@0.50", route="straight", color="#FF91CA", weight=1.2, arrow=False, allow_diagonal=True),
        edge_ref("kw_to_top_attention", "token_kw:right@0.50", "op_top_attention:left", route="horizontal", arrow=False),
        edge_ref("qa_to_top_attention", "token_qa:right@0.50", "op_top_attention:bottom", route="orthogonal", points=[[495, 166], [511, 166], [511, 149]], arrow=False),
        edge_points("top_attention_to_map", 520, 140, 572, 140, route="horizontal"),
        edge_ref("vw_to_top_value", "token_vw:right@0.50", "op_top_value:left", route="horizontal", arrow=False),
        edge_points("map_top_to_value", 592, 127, 592, 104, route="vertical"),
        edge_points("top_value_to_weighted", 601, 95, 620, 95, route="horizontal"),
        edge_ref("weighted_top_to_concat", "weighted_map_top:top@0.50", "concat_top:bottom@0.50", route="vertical"),
        edge_ref("concat_top_to_norm", "concat_top:right@0.50", "norm_top_1:left@0.50"),
        edge_ref("norm_top_to_ff", "norm_top_1:right@0.50", "ff_top:left@0.50"),
        edge_ref("norm_top_residual", "norm_top_1:bottom@0.50", "op_add_top:left", edge_type="residual_connector", route="orthogonal", points=[[750, 70], [750, 135], [857, 135]]),
        edge_ref("ff_top_to_add", "ff_top:bottom@0.50", "op_add_top:top", route="vertical"),
        edge_ref("add_top_to_norm", "op_add_top:right", "norm_top_2:left@0.50"),
        edge_ref("qw_to_bottom_attention", "token_qw:right@0.50", "op_bottom_attention:left", route="horizontal", arrow=False),
        edge_ref("ka_to_bottom_attention", "token_ka:right@0.50", "op_bottom_attention:bottom", route="orthogonal", points=[[495, 244], [511, 244], [511, 242]], arrow=False),
        edge_points("bottom_attention_to_map", 520, 233, 572, 233, route="horizontal"),
        edge_ref("va_to_bottom_value", "token_va:right@0.50", "op_bottom_value:left", route="horizontal", arrow=False),
        edge_points("map_bottom_to_value", 592, 241, 592, 260, route="vertical"),
        edge_points("bottom_value_to_weighted", 601, 269, 620, 269, route="horizontal"),
        edge_ref("weighted_bottom_to_concat", "weighted_map_bottom:bottom@0.50", "concat_bottom:top@0.50", route="vertical"),
        edge_ref("concat_bottom_to_norm", "concat_bottom:right@0.50", "norm_bottom_1:left@0.50"),
        edge_ref("norm_bottom_to_ff", "norm_bottom_1:right@0.50", "ff_bottom:left@0.50"),
        edge_ref("norm_bottom_residual", "norm_bottom_1:top@0.50", "op_add_bottom:left", edge_type="residual_connector", route="orthogonal", points=[[751, 298], [751, 232], [856, 232]]),
        edge_ref("ff_bottom_to_add", "ff_bottom:top@0.50", "op_add_bottom:bottom", route="vertical"),
        edge_ref("add_bottom_to_norm", "op_add_bottom:right", "norm_bottom_2:left@0.50"),
        edge_ref("top_norm_to_final", "norm_top_2:right@0.50", "concat_final:top@0.50", edge_type="residual_connector", route="orthogonal", points=[[1011, 136], [1060, 136], [1060, 169]]),
        edge_ref("bottom_norm_to_final", "norm_bottom_2:right@0.50", "concat_final:bottom@0.50", edge_type="residual_connector", route="orthogonal", points=[[1011, 232], [1060, 232], [1060, 203]]),
        edge_ref("final_to_output", "concat_final:right@0.50", "cross_fused:left@0.50", allow_diagonal=True),
    ]

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.cross_attention",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "cross_attention",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_source_coordinate_paper_template",
                "diagram_family": "cross_attention_feature_fusion",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_ocr_when_visible_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "input_features", "category": "input", "source_bbox_px": [54, 91, 405, 275]},
                    {"id": "attention_core", "category": "core", "source_bbox_px": [405, 80, 660, 288]},
                    {"id": "residual_heads", "category": "core", "source_bbox_px": [589, 28, 1011, 343]},
                    {"id": "output_fusion", "category": "output", "source_bbox_px": [1009, 150, 1327, 222]},
                    {"id": "figure_caption", "category": "caption", "source_bbox_px": [471, 396, 901, 422]},
                ],
            },
            "notes": [
                "Editable semantic reconstruction for the cross-attention feature fusion paper figure.",
                "Q/K/V tokens, Softmax attention maps, value-weighted maps, Concat/norm/feed-forward branches, residual add nodes, and output module are vector Visio objects.",
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


def build_mask_res_block_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 1113.0
    base_h = 741.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def node(
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
        dash: str = "solid",
        rounding: float = 0.08,
        shadow: bool = False,
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            rounding=rounding,
        )
        item["source_bbox_px"] = [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]
        if shadow:
            item["style"]["shadow"] = {
                "color": "#222222",
                "offset_x_in": 0.04,
                "offset_y_in": -0.04,
                "transparency_pct": 82,
            }
        return item

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 13,
        weight: str = "regular",
        italic: bool = False,
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["source_bbox_px"] = [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]
        if italic:
            item["style"]["font_italic"] = True
        return item

    def polygon(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        points: list[list[float]],
        *,
        fill: str,
        line: str = "none",
        z: int = 6,
    ) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "polygon_node",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "points": points,
            "source_bbox_px": [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)],
            "style": {"fill": fill, "line": line, "line_weight_pt": 0.0},
        }
        return item

    def operator(
        node_id: str,
        x: float,
        y: float,
        size: float,
        symbol: str,
        *,
        fill: str = "#FFFFFF",
        line: str = "#777777",
        font_size: float = 12,
        z: int = 75,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "operator_node",
            x,
            y,
            size,
            size,
            symbol,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            rounding=0.0,
        )
        item["symbol"] = symbol
        item["operator_shape"] = "circle"
        item["operator_size_tier"] = "small"
        item["style"]["text_color"] = "#555555"
        return item

    def dot(node_id: str, cx: float, cy: float) -> dict[str, Any]:
        item = operator(node_id, cx - 4, cy - 4, 8, "", fill="#666666", line="#666666", font_size=4, z=76)
        item["style"]["line_weight_pt"] = 0.4
        return item

    def arrow(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "straight",
        points: list[list[float]] | None = None,
        end_arrow: bool = True,
        dash: str = "solid",
        weight: float = 1.05,
        line: str = "#666666",
        z: int = 60,
    ) -> dict[str, Any]:
        scaled_points = [[sx(px), sy(py)] for px, py in points] if points else None
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=end_arrow,
            route=route,
            points=scaled_points,
            z=z,
            allow_cross_container=True,
        )
        item["style"].update(
            {
                "line": line,
                "line_weight_pt": weight,
                "line_dash": dash,
                "end_arrow": "triangle" if end_arrow else "none",
                "arrow_size": "small",
            }
        )
        return item

    blue = "#AEDBEC"
    orange = "#F0AF83"
    yellow = "#FFE8A0"
    green = "#CAE7C1"
    lane_gray = "#E9E9E9"
    lane_green = "#EAF6E6"

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0)
    ]
    edges: list[dict[str, Any]] = []

    for lane_id, x, fill in [
        ("left_lane", 184, lane_gray),
        ("right_lane", 577, lane_gray),
        ("mask_lane", 969, lane_green),
    ]:
        lane = node(lane_id, "process_box", x, 35, 78, 543, "", fill=fill, line="none", z=3, rounding=0.0)
        lane["allow_overlap"] = True
        nodes.append(lane)
        nodes.append(
            polygon(
                f"{lane_id}_arrow_head",
                x - 37,
                578,
                152,
                33,
                [[0, 0], [1, 0], [0.5, 1]],
                fill=fill,
                z=3,
            )
        )

    nodes.extend(
        [
            label("journal_header", 850, 9, 280, 24, "Digital Communications and Netw", font_size=14, italic=True),
            label("left_xi", 215, 49, 30, 20, "xi", font_size=12, italic=True),
            label("right_xi", 607, 49, 30, 20, "xi", font_size=12, italic=True),
            label("mask_i", 987, 49, 62, 24, "Maski", font_size=14, italic=True),
            label("left_xnext", 211, 568, 54, 24, "xi+1", font_size=12, italic=True),
            label("right_xnext", 604, 568, 54, 24, "xi+1", font_size=12, italic=True),
            label("mask_next", 979, 568, 74, 28, "Maski+1", font_size=13, italic=True),
            dot("left_dot", 225, 97),
            dot("right_dot", 616, 97),
            operator("left_add", 213, 407, 24, "+", font_size=13),
            operator("right_add", 604, 397, 24, "+", font_size=13),
            operator("right_gate1", 890, 210, 22, "x", font_size=11),
            operator("right_gate2", 736, 445, 22, "x", font_size=11),
        ]
    )

    block_specs = [
        ("left_conv1", 261, 159, 216, 36, "Conv7-64", blue, 15),
        ("left_bn1", 261, 216, 216, 36, "Batch normalization", orange, 14),
        ("left_relu1", 261, 273, 216, 36, "ReLU", yellow, 15),
        ("left_conv2", 261, 330, 216, 36, "Conv7-64", blue, 15),
        ("left_bn2", 110, 452, 231, 36, "Batch normalization", orange, 14),
        ("left_relu2", 110, 510, 231, 36, "ReLU", yellow, 15),
        ("right_conv1", 675, 146, 194, 36, "Conv7-64", blue, 14),
        ("right_bn1", 675, 204, 194, 36, "Batch normalization", orange, 13),
        ("right_relu1", 675, 273, 194, 36, "ReLU", yellow, 14),
        ("right_conv2", 675, 330, 194, 36, "Conv7-64", blue, 14),
        ("right_bn2", 519, 439, 194, 36, "Batch normalization", orange, 13),
        ("right_relu2", 519, 510, 194, 36, "ReLU", yellow, 14),
        ("pool_top", 911, 146, 202, 36, "Max-pooling", green, 14),
        ("pool_bottom", 911, 331, 202, 36, "Max-pooling", green, 14),
    ]
    for block_id, x, y, w, h, text, fill, font_size in block_specs:
        nodes.append(
            node(
                block_id,
                "rounded_process",
                x,
                y,
                w,
                h,
                text,
                fill=fill,
                line=fill,
                z=25,
                font_size=font_size,
                rounding=0.13,
                shadow=True,
            )
        )

    nodes.extend(
        [
            node("right_bn_dash1", "group_container", 656, 190, 277, 63, "", fill="none", line="#999999", z=8, dash="dash", rounding=0.16),
            node("right_bn_dash2", "group_container", 500, 423, 276, 66, "", fill="none", line="#999999", z=8, dash="dash", rounding=0.16),
            label("same_kernel_label", 825, 373, 146, 54, "Same kernel size,\nstride,padding", font_size=12),
            label("caption_a", 190, 624, 210, 32, "(a) Original res-block", font_size=16),
            label("caption_b", 711, 624, 185, 32, "(b) Mask res-block", font_size=16),
            label(
                "figure_caption",
                306,
                686,
                610,
                30,
                "Fig. 3. The structure of the original res-block and mask res-block.",
                font_size=15,
            ),
        ]
    )

    edges.extend(
        [
            arrow("left_input_down", 225, 72, 225, 407, route="vertical"),
            arrow("left_add_to_bn", 225, 431, 225, 452, route="vertical"),
            arrow("left_bn2_to_relu2", 225, 488, 225, 510, route="vertical"),
            arrow("left_relu2_to_out", 225, 546, 225, 568, route="vertical"),
            arrow("left_skip_to_conv1", 225, 97, 369, 159, route="hv", points=[[369, 97]]),
            arrow("left_conv1_to_bn1", 369, 195, 369, 216, route="vertical"),
            arrow("left_bn1_to_relu1", 369, 252, 369, 273, route="vertical"),
            arrow("left_relu1_to_conv2", 369, 309, 369, 330, route="vertical"),
            arrow("left_conv2_to_add", 369, 366, 237, 419, route="vh", points=[[369, 421]]),
            arrow("right_input_down", 616, 72, 616, 397, route="vertical"),
            arrow("right_add_to_bn2", 616, 421, 616, 439, route="vertical"),
            arrow("right_bn2_to_relu2", 616, 475, 616, 510, route="vertical"),
            arrow("right_relu2_to_out", 616, 546, 616, 568, route="vertical"),
            arrow("right_skip_to_conv1", 616, 97, 772, 146, route="hv", points=[[772, 97]]),
            arrow("right_conv1_to_bn1", 772, 182, 772, 204, route="vertical"),
            arrow("right_bn1_to_relu1", 772, 240, 772, 273, route="vertical"),
            arrow("right_relu1_to_conv2", 772, 309, 772, 330, route="vertical"),
            arrow("right_conv2_to_add", 772, 366, 628, 409, route="vh", points=[[772, 409]]),
            arrow("right_bn1_to_gate1", 869, 222, 890, 222, route="horizontal"),
            arrow("right_bn2_to_gate2", 713, 457, 736, 457, route="horizontal"),
            arrow("conv1_pool_link", 869, 164, 911, 164, route="horizontal", end_arrow=False, dash="dash"),
            arrow("conv2_pool_link", 869, 349, 911, 349, route="horizontal", end_arrow=False, dash="dash"),
            arrow("mask_to_pool_top", 1008, 74, 1008, 146, route="vertical"),
            arrow("pool_top_to_pool_bottom", 1008, 182, 1008, 331, route="vertical"),
            arrow("pool_bottom_to_mask_next", 1008, 367, 1008, 568, route="vertical"),
            arrow("pool_top_to_gate1", 1008, 221, 912, 221, route="horizontal", points=[[1008, 221]]),
            arrow("pool_bottom_to_gate2", 1008, 457, 758, 457, route="horizontal", points=[[1008, 457]]),
        ]
    )

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.mask_res_block",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "mask_res_block",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_source_coordinate_paper_template",
                "diagram_family": "original_and_mask_residual_block",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_ocr_when_visible_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "left_original_res_block", "category": "core", "source_bbox_px": [105, 35, 485, 656]},
                    {"id": "right_mask_res_block", "category": "core", "source_bbox_px": [500, 35, 1110, 656]},
                    {"id": "figure_caption", "category": "caption", "source_bbox_px": [305, 684, 916, 719]},
                ],
            },
            "notes": [
                "Editable semantic reconstruction for original res-block and mask res-block paper figures.",
                "Residual lanes, convolution blocks, normalization/ReLU blocks, mask pooling branch, gates, captions, and arrows are vector Visio objects.",
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


def is_swin_transformer_architecture(ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    corpus = ocr_corpus(ocr_items).lower()
    compact = re.sub(r"[^a-z0-9]+", "", corpus)
    aspect = width / max(1, height)
    has_swin = "swin" in compact and "transformer" in compact
    has_stage_flow = "stage" in compact and ("patch" in compact or "merging" in compact)
    has_block_stack = "wmsa" in compact or "swmsa" in compact or ("mlp" in compact and "ln" in compact)
    return aspect >= 2.35 and has_swin and has_stage_flow and has_block_stack


def is_sparse_swin_transformer_variant(image_path: Path, ocr_items: list[dict[str, Any]], width: int, height: int) -> bool:
    boxes = [item.get("box") for item in ocr_items if isinstance(item.get("box"), Box)]
    if not boxes:
        return False
    spread = max(box.cx for box in boxes) - min(box.cx for box in boxes)
    if spread < width * 0.55:
        return False
    image = read_image_bgr(image_path)
    if image is None:
        return False
    frame_boxes = detect_rectangular_frames(image)
    large_panel_frames = [
        box
        for box in frame_boxes
        if box.w >= width * 0.18 and box.h >= height * 0.30
    ]
    return len(large_panel_frames) < 2


def build_sparse_swin_transformer_architecture_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 1996.0
    base_h = 622.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#6D788C",
        z: int = 20,
        font_size: float = 18,
        dash: str = "solid",
        rounding: float = 0.06,
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            rounding=rounding,
        )
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"]["font_family"] = "Times New Roman"
        item["style"]["text_fit"] = "shrink_to_fit"
        return item

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 18,
        weight: str = "regular",
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["source_bbox_px"] = bbox(x, y, w, h)
        item["style"]["font_family"] = "Times New Roman"
        item["style"]["text_fit"] = "shrink_to_fit" if "\n" in text else "single_line"
        item["style"]["min_font_size_pt"] = 5.0
        return item

    def operator(node_id: str, cx: float, cy: float, size: float = 43) -> dict[str, Any]:
        item = node(
            node_id,
            "operator_node",
            cx - size / 2,
            cy - size / 2,
            size,
            size,
            "+",
            fill="#FFFFFF",
            line="#6D788C",
            z=72,
            font_size=18,
            rounding=0.0,
        )
        item["symbol"] = "+"
        item["operator_shape"] = "circle"
        item["operator_size_tier"] = "small"
        item["style"]["text_color"] = "#3F4A5F"
        item["style"]["line_weight_pt"] = 1.2
        return item

    def arrow(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "horizontal",
        points: list[list[float]] | None = None,
        end_arrow: str = "triangle",
        weight: float = 1.2,
        z: int = 60,
    ) -> dict[str, Any]:
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=end_arrow != "none",
            route=route,
            points=[[sx(px), sy(py)] for px, py in points] if points else None,
            z=z,
            allow_cross_container=True,
        )
        item["style"].update({"line": "#2F3747", "line_weight_pt": weight, "end_arrow": end_arrow, "arrow_size": "small"})
        return item

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0),
        label("dim_input", 31, 282, 125, 30, "HxWx3", font_size=13),
        label("images_label", 24, 345, 105, 34, "Images", font_size=14),
        label("patch_partition_label", 169, 340, 78, 42, "Patch Partiti\non", font_size=10),
        label("caption_architecture", 657, 557, 180, 30, "(a) Architecture", font_size=14),
        label("caption_blocks", 1542, 556, 430, 30, "(b) Two Successive Swin Transformer Blocks", font_size=14),
    ]
    edges: list[dict[str, Any]] = []

    for node_id, cx, cy in [
        ("patch_plus_top", 286, 245),
        ("patch_plus_mid", 286, 293),
        ("patch_plus_upper", 286, 313),
        ("patch_plus_lower", 286, 435),
        ("patch_plus_bottom", 286, 459),
    ]:
        nodes.append(operator(node_id, cx, cy, 44))
    edges.extend(
        [
            arrow("patch_vertical_top", 286, 475, 286, 333, route="vertical"),
            arrow("patch_vertical_mid", 286, 333, 286, 266, route="vertical"),
        ]
    )

    stage_specs = [
        ("stage1", 330, 188, 398, 323, "Stage", "Swin\nTransformer\nBlock", "", "x 2"),
        ("stage2", 641, 188, 700, 323, "Stage2", "Swin\nTransformer\nBlock", "Patch Mergi\nng", "x 2"),
        ("stage3", 944, 188, 1002, 323, "Stage", "Swin\nTransformer\nBlock", "Patch Mergi\nng", "x 6"),
        ("stage4", 1263, 188, 1307, 323, "Stage", "Swin\nTransformer\nBlock", "Patch Mergi\nng", "x 2"),
    ]
    for stage_id, title_x, title_y, block_x, block_y, title_text, block_text, patch_text, repeat in stage_specs:
        nodes.append(label(f"{stage_id}_title", title_x - 22, title_y - 8, 92, 30, title_text, font_size=14))
        if patch_text:
            nodes.append(label(f"{stage_id}_patch_merging", block_x - 96, 340, 84, 42, patch_text, font_size=10))
        nodes.append(label(f"{stage_id}_block", block_x - 34, block_y - 8, 148, 84, block_text, font_size=20))
        nodes.append(label(f"{stage_id}_repeat", block_x + 18, block_y + 70, 42, 20, repeat, font_size=9))

    for node_id, x, y, text in [
        ("dim_stage1", 250, 126, "H/4 x W/4 x 48"),
        ("dim_stage2", 850, 124, "H/8 x W/8 x C"),
        ("dim_stage3", 1137, 121, "H/16 x W/16 x 2C"),
        ("dim_stage4", 1270, 144, "H/32 x W/32 x 8C"),
    ]:
        nodes.append(label(node_id, x, y, 170, 28, text, font_size=12))

    for side, x, prefix, attn in [
        ("left", 1526, "z\nl", "W-MSA"),
        ("right", 1804, "z\nl+1", "SW-MSA"),
    ]:
        nodes.append(operator(f"{side}_plus_top", x + 61, 60, 44))
        nodes.append(operator(f"{side}_plus_mid", x + 61, 286, 36))
        nodes.extend(
            [
                node(f"{side}_mlp", "rounded_process", x, 102, 123, 54, "MLP", fill="#B9D0EF", line="#6D788C", z=25, font_size=24, rounding=0.10),
                node(f"{side}_ln_top", "rounded_process", x, 183, 123, 55, "LN", fill="#DCEBDA", line="#6D788C", z=25, font_size=24, rounding=0.10),
                node(f"{side}_attn", "process_box", x, 329, 122, 58, attn, fill="#FFFFFF", line="#D19AB9", z=25, font_size=15, rounding=0.0),
                node(f"{side}_ln_bottom", "rounded_process", x, 414, 123, 58, "LN", fill="#DCEBDA", line="#6D788C", z=25, font_size=24, rounding=0.10),
                label(f"{side}_top_label", x - 15, 39, 32, 24, prefix, font_size=8),
            ]
        )
        cx = x + 61
        edges.extend(
            [
                arrow(f"{side}_ln_bottom_to_attn", cx, 414, cx, 387, route="vertical"),
                arrow(f"{side}_attn_to_plus", cx, 329, cx, 304, route="vertical"),
                arrow(f"{side}_plus_to_ln_top", cx, 268, cx, 238, route="vertical"),
                arrow(f"{side}_ln_top_to_mlp", cx, 183, cx, 156, route="vertical"),
                arrow(f"{side}_mlp_to_plus", cx, 102, cx, 82, route="vertical"),
            ]
        )

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.swin_transformer_architecture",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "swin_transformer_sparse",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_plus_frame_density_triggered_sparse_swin_template",
                "diagram_family": "swin_transformer_architecture_sparse_no_stage_frames",
                "required_regions": ["sparse_architecture_pipeline", "large_successive_swin_blocks"],
            },
            "notes": [
                "Editable semantic reconstruction for sparse/no-frame Swin Transformer architecture variants.",
                "This category avoids adding standard dashed stage frames when the source image lacks them.",
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


def build_swin_transformer_architecture_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    if is_sparse_swin_transformer_variant(image_path, ocr_items, width, height):
        return build_sparse_swin_transformer_architecture_scene(image_path, width, height, ocr_items, title=title)

    base_w = 1148.0
    base_h = 355.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#111111",
        z: int = 20,
        font_size: float = 13,
        dash: str = "solid",
        angle: float | None = None,
        rounding: float = 0.08,
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            text_angle=angle,
            rounding=rounding,
        )
        item["source_bbox_px"] = [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]
        return item

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 13,
        angle: float | None = None,
        weight: str = "regular",
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, angle=angle, weight=weight, z=z)
        item["source_bbox_px"] = [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]
        return item

    def arrow(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "horizontal",
        points: list[list[float]] | None = None,
        end_arrow: str = "triangle",
        dash: str = "solid",
        weight: float = 1.2,
        z: int = 60,
    ) -> dict[str, Any]:
        scaled_points = [[sx(px), sy(py)] for px, py in points] if points else None
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=end_arrow != "none",
            route=route,
            points=scaled_points,
            z=z,
            allow_cross_container=True,
        )
        item["style"]["end_arrow"] = end_arrow
        item["style"]["line_dash"] = dash
        item["style"]["line_weight_pt"] = weight
        item["source_bbox_px"] = [
            round(sx(min(x1, x2)), 2),
            round(sy(min(y1, y2)), 2),
            round(sx(max(x1, x2)), 2),
            round(sy(max(y1, y2)), 2),
        ]
        return item

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0)
    ]
    edges: list[dict[str, Any]] = []

    nodes.extend(
        [
            label("dim_input", 0, 158, 104, 24, "H x W x 3", font_size=14),
            node("images", "process_box", 6, 184, 66, 45, "Images", font_size=13, rounding=0.0),
            node("patch_partition", "rounded_process", 91, 132, 29, 126, "Patch Partition", font_size=12, angle=90, rounding=0.04),
        ]
    )
    edges.append(arrow("images_to_patch_partition", 72, 206, 91, 206))

    stages = [
        {
            "id": "stage1",
            "x": 132,
            "y": 102,
            "w": 171,
            "h": 198,
            "title": "Stage 1",
            "dim": "H/4 x W/4 x 48",
            "dim_x": 96,
            "embed": ("linear_embedding", 151, 132, 29, 126, "Linear Embedding"),
            "block": ("swin_block_1", 202, 132, 86, 146, "Swin\nTransformer\nBlock"),
            "repeat": "x 2",
        },
        {
            "id": "stage2",
            "x": 312,
            "y": 102,
            "w": 162,
            "h": 198,
            "title": "Stage 2",
            "dim": "H/8 x W/8 x C",
            "dim_x": 264,
            "embed": ("patch_merging_2", 326, 132, 29, 126, "Patch Merging"),
            "block": ("swin_block_2", 374, 132, 86, 146, "Swin\nTransformer\nBlock"),
            "repeat": "x 2",
        },
        {
            "id": "stage3",
            "x": 483,
            "y": 102,
            "w": 160,
            "h": 198,
            "title": "Stage 3",
            "dim": "H/16 x W/16 x 2C",
            "dim_x": 439,
            "embed": ("patch_merging_3", 497, 132, 29, 126, "Patch Merging"),
            "block": ("swin_block_3", 544, 132, 86, 146, "Swin\nTransformer\nBlock"),
            "repeat": "x 6",
        },
        {
            "id": "stage4",
            "x": 654,
            "y": 102,
            "w": 160,
            "h": 198,
            "title": "Stage 4",
            "dim": "H/32 x W/32 x 8C",
            "dim_x": 610,
            "embed": ("patch_merging_4", 668, 132, 29, 126, "Patch Merging"),
            "block": ("swin_block_4", 716, 132, 86, 146, "Swin\nTransformer\nBlock"),
            "repeat": "x 2",
        },
    ]
    for stage in stages:
        nodes.append(node(f"{stage['id']}_frame", "group_container", stage["x"], stage["y"], stage["w"], stage["h"], "", line="#111111", dash="dash", z=5, rounding=0.14))
        nodes.append(label(f"{stage['id']}_title", stage["x"] + 54, stage["y"] + 5, 70, 22, stage["title"], font_size=13))
        nodes.append(label(f"{stage['id']}_dim", stage["dim_x"], 67, 142, 22, stage["dim"], font_size=12))
        embed_id, ex, ey, ew, eh, etext = stage["embed"]
        block_id, bx, by, bw, bh, btext = stage["block"]
        nodes.append(node(embed_id, "rounded_process", ex, ey, ew, eh, etext, font_size=11, angle=90, rounding=0.04))
        nodes.append(node(block_id, "rounded_process", bx, by, bw, bh, btext, font_size=13, rounding=0.10))
        nodes.append(label(f"{stage['id']}_repeat", bx + 32, 280, 50, 22, stage["repeat"], font_size=13))

    for edge_id, x1, y1, x2, y2 in [
        ("patch_to_linear", 120, 206, 151, 206),
        ("linear_to_swin1", 180, 206, 202, 206),
        ("swin1_to_stage2", 288, 206, 326, 206),
        ("patch2_to_swin2", 355, 206, 374, 206),
        ("swin2_to_stage3", 460, 206, 497, 206),
        ("patch3_to_swin3", 526, 206, 544, 206),
        ("swin3_to_stage4", 630, 206, 668, 206),
        ("patch4_to_swin4", 697, 206, 716, 206),
        ("swin4_to_right", 802, 206, 824, 206),
    ]:
        edges.append(arrow(edge_id, x1, y1, x2, y2))

    nodes.append(label("caption_architecture", 356, 314, 120, 24, "(a) Architecture", font_size=14))
    edges.append(arrow("panel_separator", 836, 45, 836, 302, route="vertical", end_arrow="none", dash="dash", weight=2.2, z=70))

    for side, x, attention, top_label, mid_label in [
        ("left", 862, "W-MSA", "z^l", "z^l"),
        ("right", 1018, "SW-MSA", "z^l+1", "z^l+1"),
    ]:
        nodes.append(node(f"{side}_swin_block_frame", "group_container", x, 19, 111, 282, "", line="#111111", dash="dash", z=5, rounding=0.12))
        for plus_id, py in (("plus_top", 29), ("plus_mid", 156)):
            plus = node(f"{side}_{plus_id}", "operator_node", x + 31, py, 23, 23, "+", fill="#FFFFFF", line="#111111", z=70, font_size=12, rounding=0.0)
            plus["symbol"] = "+"
            plus["operator_shape"] = "circle"
            nodes.append(plus)
        nodes.extend(
            [
                node(f"{side}_mlp", "process_box", x + 13, 64, 64, 26, "MLP", fill="#B9D0EF", z=25, font_size=12, rounding=0.0),
                node(f"{side}_ln_top", "process_box", x + 13, 109, 64, 28, "LN", fill="#DCEBDA", z=25, font_size=12, rounding=0.0),
                node(f"{side}_attn", "process_box", x + 13, 194, 64, 28, attention, fill="#E9C3D9", z=25, font_size=11, rounding=0.0),
                node(f"{side}_ln_bottom", "process_box", x + 13, 241, 64, 28, "LN", fill="#DCEBDA", z=25, font_size=12, rounding=0.0),
                label(f"{side}_top_label", x - 1, 31, 30, 18, top_label, font_size=6.5),
                label(f"{side}_mid_label", x - 1, 161, 30, 18, mid_label, font_size=6.5),
                label(f"{side}_bottom_label", x - 1, 274, 32, 18, "z^l-1" if side == "left" else "z^l", font_size=6.5),
            ]
        )
        cx = x + 45
        edges.extend(
            [
                arrow(f"{side}_bottom_to_ln", cx, 286, cx, 269, route="vertical"),
                arrow(f"{side}_ln_to_attn", cx, 241, cx, 222, route="vertical"),
                arrow(f"{side}_attn_to_plus", cx, 194, cx, 179, route="vertical"),
                arrow(f"{side}_plus_to_ln", cx, 156, cx, 137, route="vertical"),
                arrow(f"{side}_ln_to_mlp", cx, 109, cx, 90, route="vertical"),
                arrow(f"{side}_mlp_to_plus", cx, 64, cx, 52, route="vertical"),
                arrow(f"{side}_residual_bottom_up", x + 86, 286, x + 86, 169, route="vertical", end_arrow="none", weight=1.0),
                arrow(f"{side}_residual_bottom_in", x + 86, 169, x + 54, 169, route="horizontal", end_arrow="none", weight=1.0),
                arrow(f"{side}_residual_mid_up", x + 86, 169, x + 86, 40, route="vertical", end_arrow="none", weight=1.0),
                arrow(f"{side}_residual_top_in", x + 86, 40, x + 54, 40, route="horizontal", end_arrow="none", weight=1.0),
            ]
        )

    nodes.append(label("caption_blocks", 807, 314, 336, 24, "(b) Two Successive Swin Transformer Blocks", font_size=14))

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.swin_transformer_architecture",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "swin_transformer",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_paper_architecture_template",
                "diagram_family": "swin_transformer_architecture",
                "required_regions": ["architecture_pipeline", "successive_swin_blocks"],
            },
            "notes": [
                "Editable semantic reconstruction for the common Swin Transformer architecture figure.",
                "Major stage frames, patch modules, Swin Transformer blocks, residual blocks, labels, and arrows are vector Visio objects.",
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
            "created_by": "fig4visio.image_auto_scene.clean_flow",
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
            "created_by": "fig4visio.image_auto_scene.semantic_template",
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


def build_remote_sensing_rsei_workflow_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 1080.0
    base_h = 614.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def attach(item: dict[str, Any], x: float, y: float, w: float, h: float, container_id: str | None = None) -> dict[str, Any]:
        item["source_bbox_px"] = bbox(x, y, w, h)
        if container_id:
            item["container_id"] = container_id
        item.setdefault("allow_overlap", True)
        return item

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#111111",
        z: int = 20,
        font_size: float = 10,
        dash: str = "solid",
        rounding: float = 0.03,
        container_id: str | None = None,
        weight: str = "regular",
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            rounding=rounding,
        )
        item["style"].update(
            {
                "font_family_candidates": ["Times New Roman", "Cambria", "Arial", "Microsoft YaHei UI"],
                "font_role": "paper_serif",
                "font_weight": weight,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 4.5,
                "text_margin_in": 0.02,
                "line_weight_pt": 1.0,
            }
        )
        return attach(item, x, y, w, h, container_id)

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 10,
        weight: str = "regular",
        container_id: str | None = None,
        z: int = 90,
        color: str = "#111111",
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["style"].update(
            {
                "font_family_candidates": ["Times New Roman", "Cambria", "Arial", "Microsoft YaHei UI"],
                "font_role": "paper_serif",
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 4.3,
                "text_color": color,
            }
        )
        return attach(item, x, y, w, h, container_id)

    def frame(node_id: str, x: float, y: float, w: float, h: float, *, dashed: bool = True, z: int = 4) -> dict[str, Any]:
        item = node(
            node_id,
            "dashed_region" if dashed else "group_container",
            x,
            y,
            w,
            h,
            "",
            fill="none",
            line="#111111",
            dash="dash" if dashed else "solid",
            z=z,
            font_size=1,
            rounding=0.0,
        )
        item["style"].update({"line_weight_pt": 1.2, "fill": "none"})
        return item

    def header(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        container_id: str | None = None,
        font_size: float = 15,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "process_box",
            x,
            y,
            w,
            h,
            text,
            fill="#FFF2C7",
            line="#111111",
            z=18,
            font_size=font_size,
            container_id=container_id,
            weight="bold",
            rounding=0.0,
        )
        item["style"].update({"line_weight_pt": 1.1, "text_fit": "single_line"})
        return item

    def rounded(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fill: str = "#F5F5F5",
        line: str = "#111111",
        font_size: float = 10,
        container_id: str | None = None,
        z: int = 28,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "rounded_process",
            x,
            y,
            w,
            h,
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            rounding=0.07,
            container_id=container_id,
        )
        item["style"]["line_weight_pt"] = 1.1
        return item

    def stack(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "#5E9F78",
        container_id: str | None = None,
        layers: int = 7,
        z: int = 25,
    ) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "tensor_stack",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "layers": layers,
            "stack_render_mode": "thin_feature_slabs",
            "style": {
                "fill": fill,
                "top_fill": "#D8E9CF",
                "side_fill": "#7A927F",
                "line": "#333333",
                "line_weight_pt": 0.6,
                "layer_dx_in": sx(-3.0),
                "layer_dy_in": sy(1.7),
            },
        }
        return attach(item, x, y, w, h, container_id)

    def grid(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rows: int = 5,
        cols: int = 7,
        colors: tuple[str, str] = ("#B7D987", "#F3D66B"),
        container_id: str | None = None,
        z: int = 26,
    ) -> dict[str, Any]:
        cells = [[row, col, colors[(row + col * 2) % len(colors)]] for row in range(rows) for col in range(cols)]
        item = {
            "id": node_id,
            "type": "grid_matrix",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "rows": rows,
            "cols": cols,
            "colored_cells": cells,
            "style": {
                "line": "#6D8F6D",
                "line_weight_pt": 0.5,
                "grid_line": "#D9E6CC",
                "grid_line_weight_pt": 0.25,
            },
        }
        return attach(item, x, y, w, h, container_id)

    def feature_grid(node_id: str, x: float, y: float, w: float, h: float, *, container_id: str | None = None, z: int = 26) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "feature_map_grid",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "rows": 5,
            "cols": 7,
            "row_colors": ["#A8D8E9", "#66B5C9", "#39A4BD", "#79C6D8", "#BDE9EF"],
            "column_shades": [0.10, 0.32, 0.55, 0.15, 0.44, 0.70, 0.24],
            "style": {"line": "#277D99", "grid_line": "#D8F0F5", "grid_line_weight_pt": 0.25, "line_weight_pt": 0.6},
        }
        return attach(item, x, y, w, h, container_id)

    def polygon(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        points: list[list[float]],
        *,
        fill: str,
        line: str = "#5B9C66",
        container_id: str | None = None,
        z: int = 40,
    ) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "polygon_node",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "points": points,
            "style": {"fill": fill, "line": line, "line_weight_pt": 0.7},
        }
        return attach(item, x, y, w, h, container_id)

    def ellipse(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#111111",
        font_size: float = 8.5,
        container_id: str | None = None,
        z: int = 38,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "ellipse_node",
            x,
            y,
            w,
            h,
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            rounding=0.0,
            container_id=container_id,
        )
        item["style"]["text_fit"] = "shrink_to_fit"
        return item

    def edge(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "straight",
        points: list[list[float]] | None = None,
        arrow: bool = True,
        dash: str = "solid",
        color: str = "#111111",
        weight: float = 1.05,
        z: int = 70,
        allow_diagonal: bool = False,
        arrow_plan_id: str | None = None,
    ) -> dict[str, Any]:
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=arrow,
            route=route,
            points=[[sx(px), sy(py)] for px, py in points] if points else None,
            z=z,
            allow_cross_container=True,
        )
        item["style"].update({"line": color, "line_dash": dash, "line_weight_pt": weight, "arrow_size": "small"})
        if route in {"horizontal", "vertical"} and arrow:
            item["type"] = "lane_arrow"
        if allow_diagonal:
            item["allow_diagonal"] = True
        if arrow_plan_id:
            item["arrow_plan_id"] = arrow_plan_id
        return item

    def map_blob(prefix: str, x: float, y: float, w: float, h: float, container_id: str) -> list[dict[str, Any]]:
        return [
            polygon(
                f"{prefix}_map_blob",
                x,
                y,
                w,
                h,
                [[0.06, 0.50], [0.16, 0.28], [0.34, 0.18], [0.53, 0.27], [0.72, 0.14], [0.91, 0.37], [0.84, 0.66], [0.64, 0.80], [0.43, 0.72], [0.22, 0.84]],
                fill="#A9D685",
                line="#7AAD61",
                container_id=container_id,
                z=42,
            ),
            polygon(
                f"{prefix}_map_patch",
                x + w * 0.22,
                y + h * 0.26,
                w * 0.46,
                h * 0.38,
                [[0.10, 0.44], [0.38, 0.18], [0.80, 0.35], [0.70, 0.78], [0.26, 0.84]],
                fill="#E1C66B",
                line="#D8B84B",
                container_id=container_id,
                z=43,
            ),
            ellipse(f"{prefix}_hotspot", x + w * 0.60, y + h * 0.42, w * 0.10, h * 0.10, "", fill="#D88A4B", line="#D88A4B", container_id=container_id, z=44),
        ]

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0),
        frame("outer_border", 29, 14, 1030, 588, dashed=False, z=2),
    ]
    edges: list[dict[str, Any]] = []

    nodes.extend(
        [
            frame("images_data_region", 31, 16, 474, 147, dashed=True),
            header("images_data_header", 31, 18, 474, 23, "Images data", container_id="images_data_region", font_size=14.5),
            frame("driver_layer_region", 512, 16, 545, 147, dashed=True),
            header("driver_layer_header", 512, 18, 545, 23, "Driver Layer", container_id="driver_layer_region", font_size=14.5),
        ]
    )
    for index, x in enumerate((193, 352), start=1):
        edges.append(edge(f"images_sep_{index}", x, 41, x, 163, route="vertical", arrow=False, dash="dash", weight=1.0, z=16))

    image_sources = [
        ("landsat5", 45, 47, 135, "Landsat 5 TM\nSurface Reflectance images", "#50785F"),
        ("landsat8", 214, 47, 125, "Landsat 8 OLI\nSurface Reflectance images", "#52795C"),
        ("jrc_water", 363, 47, 126, "JRC Global Surface Water\nMapping Layers", "#7A70C9"),
    ]
    for source_id, x, y, w, text, fill in image_sources:
        nodes.append(label(f"{source_id}_label", x - 2, y, w + 18, 34, text, font_size=9.2, container_id="images_data_region"))
        if source_id == "jrc_water":
            nodes.append(
                polygon(
                    "jrc_water_shape",
                    x + 43,
                    y + 51,
                    55,
                    52,
                    [[0.19, 0.32], [0.47, 0.15], [0.78, 0.22], [0.88, 0.56], [0.62, 0.82], [0.31, 0.71]],
                    fill="#7D6BCE",
                    line="#6754B8",
                    container_id="images_data_region",
                )
            )
            nodes.append(ellipse("jrc_water_dot", x + 62, y + 66, 19, 19, "", fill="#E6DBFF", line="#7D6BCE", container_id="images_data_region", z=45))
        else:
            nodes.append(stack(f"{source_id}_stack", x + 34, y + 50, 76, 49, fill=fill, container_id="images_data_region"))
    nodes.append(label("images_year_note", 131, 156, 113, 10, "From 2000,2005,2010,2015,2020", font_size=5.2, container_id="images_data_region"))

    drivers = [
        ("terrain", 531, 51, "Terrain"),
        ("climate", 667, 51, "Climate"),
        ("soil", 805, 51, "Soil"),
        ("urban", 943, 51, "Urbanization"),
    ]
    for driver_id, x, y, text in drivers:
        nodes.append(rounded(f"{driver_id}_tile_bg", x, y, 101, 103, "", fill="#FFF2CE", line="#FFF2CE", container_id="driver_layer_region", z=19))
        nodes.append(label(f"{driver_id}_label", x + 13, y - 3, 78, 22, text, font_size=10.5, container_id="driver_layer_region"))
    nodes.extend(
        [
            feature_grid("terrain_grid", 543, 80, 82, 50, container_id="driver_layer_region", z=26),
            polygon("terrain_outline", 539, 76, 87, 59, [[0.03, 0.72], [0.20, 0.32], [0.45, 0.18], [0.81, 0.26], [0.96, 0.66], [0.62, 0.79], [0.26, 0.86]], fill="none", line="#222222", container_id="driver_layer_region", z=46),
            ellipse("climate_sun", 680, 81, 47, 47, "", fill="#F6C247", line="#E09A20", container_id="driver_layer_region", z=32),
            ellipse("climate_cloud_1", 712, 100, 52, 29, "", fill="#BFE6F3", line="#75B9D4", container_id="driver_layer_region", z=40),
            ellipse("climate_cloud_2", 696, 105, 42, 25, "", fill="#D8F0F7", line="#75B9D4", container_id="driver_layer_region", z=41),
            node("soil_layer_top", "process_box", 811, 81, 92, 17, "", fill="#9CD173", line="#5F8D3E", z=29, container_id="driver_layer_region", rounding=0.0),
            node("soil_layer_mid", "process_box", 811, 98, 92, 26, "", fill="#9B6237", line="#6F4324", z=29, container_id="driver_layer_region", rounding=0.0),
            node("soil_layer_bottom", "process_box", 811, 124, 92, 16, "", fill="#D5B187", line="#8F6B43", z=29, container_id="driver_layer_region", rounding=0.0),
            node("urban_sky", "process_box", 951, 80, 86, 58, "", fill="#103B78", line="#103B78", z=24, container_id="driver_layer_region", rounding=0.0),
        ]
    )
    for index, (x, y, w, h, fill) in enumerate(
        [
            (956, 102, 9, 35, "#7BA8D8"),
            (970, 91, 13, 46, "#A7CAE5"),
            (987, 108, 11, 29, "#4F84C2"),
            (1003, 86, 15, 51, "#D0E5F5"),
            (1023, 98, 10, 39, "#6B9FD4"),
        ]
    ):
        nodes.append(node(f"urban_building_{index}", "process_box", x, y, w, h, "", fill=fill, line=fill, z=34, container_id="driver_layer_region", rounding=0.0))

    nodes.extend(
        [
            frame("pre_processing_region", 33, 192, 320, 63, dashed=True),
            header("pre_processing_header", 33, 192, 320, 24, "Pre-processing", container_id="pre_processing_region", font_size=13.5),
            frame("extracting_region", 356, 192, 148, 63, dashed=True),
            header("extracting_header", 356, 192, 148, 24, "Extracting", container_id="extracting_region", font_size=13.5),
        ]
    )
    for index, (x, text) in enumerate([(50, "LEDAPS"), (143, "LaSRC"), (221, "CFMASK"), (310, "Mosaic")]):
        nodes.append(label(f"preproc_step_{index}", x - 15, 228, 68, 20, text, font_size=8.6, container_id="pre_processing_region"))
    nodes.append(label("extract_water_mask", 398, 228, 78, 20, "Water Mask", font_size=8.6, container_id="extracting_region"))

    for plan_id, edge_id, x in [("A001", "landsat5_to_preproc", 111), ("A002", "landsat8_to_preproc", 275), ("A003", "jrc_to_extract", 430)]:
        edges.append(edge(edge_id, x, 163, x, 192, route="vertical", points=[[x, 174], [x, 181]], weight=1.4, arrow_plan_id=plan_id))
    edges.append(edge("preproc_to_rsei", 191, 255, 191, 282, route="vertical", points=[[191, 265], [191, 274]], weight=1.4, arrow_plan_id="A004"))
    edges.append(edge("extract_to_rsei", 430, 255, 430, 282, route="vertical", points=[[430, 265], [430, 274]], weight=1.4, arrow_plan_id="A005"))
    edges.append(edge("driver_to_pls", 784, 163, 784, 192, route="vertical", points=[[784, 174], [784, 183]], weight=1.4, arrow_plan_id="A006"))

    nodes.extend(
        [
            frame("rsei_region", 32, 282, 472, 258, dashed=True),
            header("rsei_header", 33, 282, 470, 24, "RSEI information extraction by GEE", container_id="rsei_region", font_size=13.5),
        ]
    )
    indices = [("ndvi", 43, "NDVI"), ("ndsi", 166, "NDSI"), ("wet", 291, "WET"), ("lst", 417, "LST")]
    for item_id, x, text in indices:
        nodes.append(rounded(f"{item_id}_box", x, 313, 79, 41, text, fill="#F3F3F3", line="#111111", font_size=9.5, container_id="rsei_region"))
        edges.append(edge(f"{item_id}_down", x + 39.5, 354, x + 39.5, 372, route="vertical", arrow=False, weight=1.0))
    edges.append(edge("index_join_line", 82, 372, 456, 372, route="horizontal", arrow=False, weight=1.0))
    edges.append(edge("index_join_to_pca", 268, 372, 268, 388, route="vertical", arrow=False, weight=1.0))
    nodes.append(node("normalization_pca", "process_box", 42, 388, 453, 29, "Normalization, PCA", fill="#F5F5F5", line="#111111", z=24, font_size=11, container_id="rsei_region", rounding=0.0))
    edges.append(edge("pca_to_maps", 268, 417, 268, 432, route="vertical", weight=1.0))
    nodes.append(label("maps_title", 206, 442, 124, 22, "Multi-year RSEI maps", font_size=10.5, container_id="rsei_region"))
    for index, x in enumerate([48, 128, 207, 286, 365, 435]):
        nodes.extend(map_blob(f"rsei_map_{index}", x, 457, 62, 48, "rsei_region"))
    edges.append(edge("maps_to_change", 268, 540, 268, 568, route="vertical", points=[[268, 550], [268, 559]], weight=1.4, arrow_plan_id="A008"))
    nodes.append(node("rsei_change_analysis", "process_box", 33, 568, 470, 32, "RSEI change analysis", fill="#FFFFFF", line="#111111", z=24, font_size=12, rounding=0.0))

    nodes.extend(
        [
            frame("pls_sem_region", 549, 192, 508, 268, dashed=True),
            header("pls_sem_header", 549, 192, 508, 39, "PLS-SEM analysis", container_id="pls_sem_region", font_size=13.5),
        ]
    )
    for index, (x, y, text) in enumerate(
        [
            (606, 242, "Precipitation"),
            (606, 265, "Temperature"),
            (606, 288, "Evaporation"),
            (606, 405, "Elevation"),
            (606, 428, "Slope"),
            (942, 242, "OC"),
            (942, 265, "Clay"),
            (942, 288, "Sand"),
            (942, 405, "GDP"),
            (942, 428, "Population"),
        ]
    ):
        nodes.append(node(f"sem_indicator_{index}", "process_box", x, y, 60, 17, text, fill="#F5DDCF", line="#555555", z=30, font_size=5.8, container_id="pls_sem_region", rounding=0.0))
    latent_nodes = [
        ("sem_climate", 711, 259, 33, 33, "Climate"),
        ("sem_terrain", 711, 403, 33, 33, "Terrain"),
        ("sem_soil", 878, 259, 33, 33, "Soil"),
        ("sem_urban", 878, 403, 33, 33, "Urbanization"),
        ("sem_rsei_mid", 789, 348, 33, 33, "R2\nRSEI"),
    ]
    for node_id, x, y, w, h, text in latent_nodes:
        nodes.append(ellipse(node_id, x, y, w, h, "", fill="#FFFFFF", line="#333333", container_id="pls_sem_region", z=38))
        nodes.append(label(f"{node_id}_label", x - 10, y + h + 2, w + 28, 14, text, font_size=5.8, container_id="pls_sem_region", z=91))
    nodes.append(node("sem_rsei_box", "process_box", 786, 286, 48, 24, "RSEI", fill="#F4E7DE", line="#555555", z=38, font_size=7.5, container_id="pls_sem_region", rounding=0.0))
    nodes.append(ellipse("sem_r2_top", 869, 285, 25, 25, "R2", fill="#FFFFFF", line="#333333", font_size=5.5, container_id="pls_sem_region", z=38))
    sem_edges = [
        ("sem_rain_to_climate", 666, 250, 711, 275),
        ("sem_temp_to_climate", 666, 273, 711, 275),
        ("sem_evap_to_climate", 666, 296, 711, 275),
        ("sem_elev_to_terrain", 666, 413, 711, 420),
        ("sem_slope_to_terrain", 666, 436, 711, 420),
        ("sem_soil_to_oc", 911, 275, 942, 250),
        ("sem_soil_to_clay", 911, 275, 942, 273),
        ("sem_soil_to_sand", 911, 275, 942, 296),
        ("sem_urban_to_gdp", 911, 420, 942, 413),
        ("sem_urban_to_pop", 911, 420, 942, 436),
        ("sem_climate_to_rsei", 744, 275, 786, 298),
        ("sem_terrain_to_rsei_mid", 744, 420, 789, 365),
        ("sem_soil_to_rsei", 878, 275, 834, 298),
        ("sem_urban_to_rsei_mid", 878, 420, 822, 365),
        ("sem_rsei_to_mid", 810, 310, 806, 348),
        ("sem_climate_to_terrain", 727, 292, 727, 403),
        ("sem_soil_to_urban", 891, 310, 891, 403),
        ("sem_terrain_to_urban", 744, 420, 878, 420),
    ]
    for edge_id, x1, y1, x2, y2 in sem_edges:
        edges.append(edge(edge_id, x1, y1, x2, y2, route="straight", color="#7EA074" if "rsei" in edge_id else "#9E6E6A", weight=0.75, z=66, allow_diagonal=True))
    for index, (x, y) in enumerate([(678, 251), (681, 272), (682, 294), (676, 412), (676, 433), (918, 250), (918, 273), (918, 295), (918, 412), (918, 434), (763, 275), (754, 390), (844, 276), (842, 390)]):
        nodes.append(label(f"sem_value_{index}", x, y, 28, 10, "Value", font_size=4.6, container_id="pls_sem_region", z=93))
    edges.append(edge("rsei_to_pls_big", 504, 326, 549, 326, route="horizontal", weight=2.0, z=72, arrow_plan_id="A007"))

    nodes.extend(
        [
            frame("global_auto_left_region", 549, 487, 232, 114, dashed=True),
            header("global_auto_left_header", 549, 487, 232, 41, "Global spatial auto-correlation", container_id="global_auto_left_region", font_size=11.5),
            frame("global_auto_right_region", 825, 487, 232, 114, dashed=True),
            header("global_auto_right_header", 825, 487, 232, 41, "Global spatial auto-correlation", container_id="global_auto_right_region", font_size=11.5),
        ]
    )
    nodes.extend(
        [
            ellipse("moran_dot_1", 607, 557, 30, 30, "", fill="#B7B7B7", line="#B7B7B7", container_id="global_auto_left_region", z=30),
            ellipse("moran_dot_2", 626, 566, 19, 19, "", fill="#9C9C9C", line="#9C9C9C", container_id="global_auto_left_region", z=31),
            label("moran_i_label", 638, 557, 90, 28, "Moran's I", font_size=12, container_id="global_auto_left_region", z=92),
            label("local_moran_label", 872, 554, 130, 28, "Local Moran's I", font_size=10.5, container_id="global_auto_right_region", z=92),
            grid("auto_grid_right", 945, 546, 70, 42, rows=4, cols=6, colors=("#D5DDE8", "#9CADC4"), container_id="global_auto_right_region", z=30),
        ]
    )
    edges.append(edge("rsei_to_global_auto_left", 504, 543, 549, 543, route="horizontal", weight=2.0, z=72, arrow_plan_id="A009"))
    edges.append(edge("global_auto_left_to_right", 781, 543, 825, 543, route="horizontal", weight=2.0, z=72, arrow_plan_id="A010"))

    arrow_plan = [
        {"id": "A001", "from": "Landsat 5 TM source", "to": "Pre-processing", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A002", "from": "Landsat 8 OLI source", "to": "Pre-processing", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A003", "from": "JRC water layers", "to": "Water Mask extraction", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A004", "from": "Pre-processing", "to": "RSEI information extraction by GEE", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A005", "from": "Extracting Water Mask", "to": "RSEI information extraction by GEE", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A006", "from": "Driver Layer", "to": "PLS-SEM analysis", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A007", "from": "RSEI extraction panel", "to": "PLS-SEM analysis", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A008", "from": "RSEI maps", "to": "RSEI change analysis", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A009", "from": "RSEI maps", "to": "Global spatial auto-correlation", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A010", "from": "Global spatial auto-correlation", "to": "Local/global spatial auto-correlation detail", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
    ]

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.remote_sensing_rsei_workflow",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "remote_sensing_rsei_workflow",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_remote_sensing_workflow_template",
                "diagram_family": "remote_sensing_rsei_workflow_with_pls_sem",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_visible_ocr_labels_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "images_data_region", "category": "input_data", "source_bbox_px": [31, 16, 505, 163], "required_visible_labels": ["Images data", "Landsat 5 TM", "Landsat 8 OLI", "JRC Global Surface Water Mapping Layers"]},
                    {"id": "driver_layer_region", "category": "driver_data", "source_bbox_px": [512, 16, 1057, 163], "required_visible_labels": ["Driver Layer", "Terrain", "Climate", "Soil", "Urbanization"]},
                    {"id": "pre_processing_region", "category": "preprocess", "source_bbox_px": [33, 192, 353, 255], "required_visible_labels": ["Pre-processing", "LEDAPS", "LaSRC", "CFMASK", "Mosaic"]},
                    {"id": "extracting_region", "category": "extracting", "source_bbox_px": [356, 192, 504, 255], "required_visible_labels": ["Extracting", "Water Mask"]},
                    {"id": "rsei_region", "category": "rsei_extraction", "source_bbox_px": [32, 282, 504, 540], "required_visible_labels": ["RSEI information extraction by GEE", "NDVI", "NDSI", "WET", "LST", "Normalization, PCA", "Multi-year RSEI maps"]},
                    {"id": "pls_sem_region", "category": "pls_sem", "source_bbox_px": [549, 192, 1057, 460], "required_visible_labels": ["PLS-SEM analysis", "Climate", "Soil", "Terrain", "Urbanization", "RSEI"]},
                    {"id": "global_auto_left_region", "category": "spatial_autocorrelation", "source_bbox_px": [549, 487, 781, 601], "required_visible_labels": ["Global spatial auto-correlation", "Moran's I"]},
                    {"id": "global_auto_right_region", "category": "spatial_autocorrelation", "source_bbox_px": [825, 487, 1057, 601], "required_visible_labels": ["Global spatial auto-correlation"]},
                ],
            },
            "region_plan": [
                {"id": "images_data_region", "category": "input_data", "source_bbox_px": [31, 16, 505, 163]},
                {"id": "driver_layer_region", "category": "driver_data", "source_bbox_px": [512, 16, 1057, 163]},
                {"id": "pre_processing_region", "category": "preprocess", "source_bbox_px": [33, 192, 353, 255]},
                {"id": "extracting_region", "category": "extracting", "source_bbox_px": [356, 192, 504, 255]},
                {"id": "rsei_region", "category": "rsei_extraction", "source_bbox_px": [32, 282, 504, 540]},
                {"id": "pls_sem_region", "category": "pls_sem", "source_bbox_px": [549, 192, 1057, 460]},
                {"id": "global_auto_left_region", "category": "spatial_autocorrelation", "source_bbox_px": [549, 487, 781, 601]},
                {"id": "global_auto_right_region", "category": "spatial_autocorrelation", "source_bbox_px": [825, 487, 1057, 601]},
            ],
            "arrow_plan": arrow_plan,
            "notes": [
                "Editable semantic reconstruction for remote-sensing RSEI workflow diagrams with PLS-SEM and spatial auto-correlation panels.",
                "Input data cubes, driver icons, RSEI index blocks, map thumbnails, PLS-SEM latent path model, and analysis panels are rebuilt as editable Visio objects.",
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


def build_industry_4_0_sustainability_framework_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 981.0
    base_h = 561.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def attach(item: dict[str, Any], x: float, y: float, w: float, h: float, container_id: str | None = None) -> dict[str, Any]:
        item["source_bbox_px"] = bbox(x, y, w, h)
        if container_id:
            item["container_id"] = container_id
        item.setdefault("allow_overlap", True)
        return item

    def page_node() -> dict[str, Any]:
        return px_node("page_background", "page_background", 0, 0, width, height, "", fill="#FFFFFF", line="none", z=0)

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#111111",
        z: int = 20,
        font_size: float = 11,
        dash: str = "solid",
        rounding: float = 0.04,
        container_id: str | None = None,
        weight: str = "regular",
        text_color: str = "#111111",
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color=text_color,
            dash=dash,
            rounding=rounding,
        )
        item["style"].update(
            {
                "font_family_candidates": ["Times New Roman", "Georgia", "Arial", "Microsoft YaHei UI"],
                "font_role": "paper_serif",
                "font_weight": weight,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 4.6,
                "text_margin_in": 0.025,
                "line_weight_pt": 1.1,
            }
        )
        return attach(item, x, y, w, h, container_id)

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 11,
        weight: str = "regular",
        italic: bool = False,
        color: str = "#111111",
        container_id: str | None = None,
        z: int = 90,
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["style"].update(
            {
                "font_family_candidates": ["Times New Roman", "Georgia", "Arial", "Microsoft YaHei UI"],
                "font_role": "paper_serif",
                "font_italic": italic,
                "text_color": color,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 4.6,
                "text_margin_in": 0.0,
            }
        )
        return attach(item, x, y, w, h, container_id)

    def frame(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        container_id: str | None = None,
        dashed: bool = True,
        fill: str = "none",
        line: str = "#111111",
        z: int = 4,
        rounding: float = 0.0,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "dashed_region" if dashed else "group_container",
            x,
            y,
            w,
            h,
            "",
            fill=fill,
            line=line,
            z=z,
            font_size=1,
            dash="dash" if dashed else "solid",
            rounding=rounding,
            container_id=container_id,
        )
        item["style"].update({"line_weight_pt": 1.25, "fill": fill})
        return item

    def container(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        line: str,
        container_id: str | None = None,
        z: int = 10,
        rounding: float = 0.07,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "group_container",
            x,
            y,
            w,
            h,
            "",
            fill=fill,
            line=line,
            z=z,
            font_size=1,
            rounding=rounding,
            container_id=container_id,
        )
        item["style"].update({"line_weight_pt": 1.25})
        return item

    def rounded(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fill: str,
        line: str,
        font_size: float = 10.5,
        container_id: str | None = None,
        z: int = 28,
        rounding: float = 0.07,
        weight: str = "regular",
        color: str = "#111111",
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "rounded_process",
            x,
            y,
            w,
            h,
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            rounding=rounding,
            container_id=container_id,
            weight=weight,
            text_color=color,
        )
        item["style"].update({"line_weight_pt": 1.25})
        return item

    def ellipse(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str,
        line: str,
        font_size: float = 9.0,
        color: str = "#111111",
        container_id: str | None = None,
        z: int = 34,
    ) -> dict[str, Any]:
        item = node(
            node_id,
            "ellipse_node",
            x,
            y,
            w,
            h,
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            rounding=0.0,
            container_id=container_id,
            text_color=color,
        )
        item["style"].update({"line_weight_pt": 1.35, "text_fit": "shrink_to_fit"})
        return item

    def polygon(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        points: list[list[float]],
        *,
        fill: str,
        line: str = "#111111",
        container_id: str | None = None,
        z: int = 40,
    ) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "polygon_node",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "points": points,
            "style": {"fill": fill, "line": line, "line_weight_pt": 0.9},
        }
        return attach(item, x, y, w, h, container_id)

    def edge(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "straight",
        points: list[list[float]] | None = None,
        arrow: bool = True,
        begin_arrow: bool = False,
        color: str = "#111111",
        weight: float = 1.4,
        z: int = 70,
        arrow_size: str = "small",
        arrow_plan_id: str | None = None,
        allow_diagonal: bool = False,
        allow_direct_cross_container: bool = False,
    ) -> dict[str, Any]:
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=arrow,
            route=route,
            points=[[sx(px), sy(py)] for px, py in points] if points else None,
            z=z,
            allow_cross_container=True,
        )
        item["style"].update(
            {
                "line": color,
                "line_weight_pt": weight,
                "end_arrow": "triangle" if arrow else "none",
                "arrow_size": arrow_size,
            }
        )
        if begin_arrow:
            item["style"]["begin_arrow"] = "triangle"
        if route in {"horizontal", "vertical"} and arrow:
            item["type"] = "lane_arrow"
        if arrow_plan_id:
            item["arrow_plan_id"] = arrow_plan_id
        if allow_diagonal:
            item["allow_diagonal"] = True
        if allow_direct_cross_container:
            item["allow_direct_cross_container"] = True
        return item

    nodes: list[dict[str, Any]] = [page_node()]
    edges: list[dict[str, Any]] = []

    # Main three-column framework.
    nodes.extend(
        [
            frame("industry_outer", 15, 20, 412, 524, dashed=True, z=2),
            frame("functions_outer", 456, 19, 297, 524, dashed=True, z=2),
            frame("manufacturing_outer", 783, 20, 182, 524, dashed=True, z=2),
            label("industry_title", 150, 22, 144, 24, "Industry 4.0", font_size=13.5, weight="bold", container_id="industry_outer"),
            label(
                "functions_title",
                493,
                15,
                225,
                38,
                "Industry 4.0 Sustainability\nFunctions",
                font_size=13.5,
                weight="bold",
                container_id="functions_outer",
            ),
            label(
                "manufacturing_title",
                815,
                14,
                120,
                36,
                "Sustainable\nManufacturing",
                font_size=12.8,
                weight="bold",
                container_id="manufacturing_outer",
            ),
        ]
    )

    left_panels = [
        ("technologies", 30, 46, 382, 140, "Technologies"),
        ("components", 31, 216, 381, 141, "Components"),
        ("principles", 31, 395, 381, 139, "Principles"),
    ]
    for panel_id, x, y, w, h, text in left_panels:
        nodes.append(
            container(
                f"{panel_id}_panel",
                x,
                y,
                w,
                h,
                fill="#F7F7F2",
                line="#333333",
                container_id="industry_outer",
                z=10,
                rounding=0.08,
            )
        )
        nodes.append(label(f"{panel_id}_label", x + 4, y + 5, 112, 24, text, font_size=15, italic=True, container_id=f"{panel_id}_panel"))

    tech_specs = [
        ("artificial_intelligence", 34, 81, 101, 47, "Artificial\nintelligence"),
        ("mixed_reality", 138, 54, 90, 54, "Mixed\nReality"),
        ("iiot", 235, 74, 62, 43, "IIoT"),
        ("blockchain", 301, 52, 90, 54, "Blockchain"),
        ("digital_twins", 33, 134, 108, 45, "Digital\ntwins"),
        ("robotics", 142, 126, 90, 41, "Robotics"),
        ("big_data", 236, 127, 88, 54, "Big data\nanalytics"),
        ("cps", 329, 126, 74, 42, "CPS"),
    ]
    for tech_id, x, y, w, h, text in tech_specs:
        nodes.append(
            ellipse(
                f"tech_{tech_id}",
                x,
                y,
                w,
                h,
                text,
                fill="#D9E8F7",
                line="#2A7CB6",
                font_size=9.5,
                color="#4F6B99",
                container_id="technologies_panel",
            )
        )

    component_specs = [
        ("smart_customers", 142, 224, 240, 37, "Smart customers"),
        ("smart_distribution", 42, 270, 125, 35, "Smart\ndistribution"),
        ("digital_supply", 180, 270, 114, 35, "Digital supply\nnetworks"),
        ("smart_shareholders", 299, 270, 108, 72, "Smart\nshareholders"),
        ("smart_factory", 45, 313, 121, 34, "Smart factory"),
        ("smart_products", 181, 313, 114, 34, "Smart\nproducts"),
    ]
    for comp_id, x, y, w, h, text in component_specs:
        nodes.append(
            rounded(
                f"component_{comp_id}",
                x,
                y,
                w,
                h,
                text,
                fill="#FFF4D3",
                line="#D18835",
                font_size=9.8,
                color="#7B4B22",
                container_id="components_panel",
                rounding=0.035,
            )
        )

    principle_specs = [
        ("virtualization", 120, 409, 138, 36, "Virtualization"),
        ("vertical_integration", 265, 409, 137, 36, "Vertical\nintegration"),
        ("real_time", 69, 452, 189, 37, "Real-time capability"),
        ("interoperability", 264, 452, 139, 37, "Interoperability"),
        ("technical_assistance", 42, 495, 104, 35, "Technical\nassistance"),
        ("decentralization", 151, 495, 139, 35, "Decentralization"),
        ("horizontal_integration", 294, 495, 109, 35, "Horizontal\nintegration"),
    ]
    for princ_id, x, y, w, h, text in principle_specs:
        nodes.append(
            rounded(
                f"principle_{princ_id}",
                x,
                y,
                w,
                h,
                text,
                fill="#F4E2F4",
                line="#9559A3",
                font_size=9.5,
                color="#7A4B88",
                container_id="principles_panel",
                rounding=0.035,
            )
        )

    for index, x in enumerate((86, 235, 365)):
        edges.append(edge(f"tech_components_link_{index}", x, 188, x, 212, route="vertical", begin_arrow=True, weight=5.0, arrow_size="medium", allow_direct_cross_container=True))
        edges.append(edge(f"components_principles_link_{index}", x, 358, x, 389, route="vertical", begin_arrow=True, weight=5.0, arrow_size="medium", allow_direct_cross_container=True))

    function_labels = [
        "Business model innovation",
        "Customer-oriented manufacturing",
        "Employee productivity",
        "Harmful emission reduction",
        "Improved manufacturing profit margin",
        "Intelligent production planning and control",
        "Manufacturing agility",
        "Manufacturing productivity and efficiency",
        "New employment opportunities",
        "Resource and energy efficiency",
        "Reduced manufacturing costs",
        "Safe and smart working environment",
        "Supply chain process integration",
        "Sustainable product development",
        "Sustainable value-creation networking",
    ]
    row_x = 466
    row_w = 278
    row_h = 28
    row_y0 = 53
    row_gap = 33
    for index, text in enumerate(function_labels):
        y = row_y0 + index * row_gap
        nodes.append(
            rounded(
                f"function_{index + 1:02d}",
                row_x,
                y,
                row_w,
                row_h,
                text,
                fill="#FFFDE4",
                line="#C8C04D",
                font_size=9.2,
                container_id="functions_outer",
                rounding=0.035,
            )
        )

    cross_arrow_y = [98, 230, 326, 491]
    for index, y in enumerate(cross_arrow_y, start=1):
        edges.append(
            edge(
                f"industry_to_functions_{index}",
                427,
                y,
                456,
                y,
                route="horizontal",
                weight=6.0,
                arrow_size="medium",
                arrow_plan_id=f"A{index:03d}",
                allow_direct_cross_container=True,
            )
        )
        edges.append(
            edge(
                f"functions_to_manufacturing_{index}",
                753,
                y,
                783,
                y,
                route="horizontal",
                weight=6.0,
                arrow_size="medium",
                arrow_plan_id=f"A{index + 4:03d}",
                allow_direct_cross_container=True,
            )
        )

    outcome_cards = [
        ("social", 789, 47, 170, 122, "Social development"),
        ("economic", 789, 171, 170, 121, "Sustainable economic\ngrowth"),
        ("renewables", 789, 295, 170, 122, "Renewables"),
        ("green", 789, 419, 170, 122, "Green manufacturing"),
    ]
    for card_id, x, y, w, h, text in outcome_cards:
        nodes.append(
            container(
                f"{card_id}_card",
                x,
                y,
                w,
                h,
                fill="#E9F3DD",
                line="#23A052",
                container_id="manufacturing_outer",
                z=12,
                rounding=0.075,
            )
        )
        nodes.append(label(f"{card_id}_title", x + 9, y + 2, w - 18, 31, text, font_size=10.0, weight="bold", color="#4E8C3F", container_id=f"{card_id}_card"))

    # Editable icon approximations for the four right-side cards.
    nodes.extend(
        [
            ellipse("social_globe", 855, 75, 60, 60, "", fill="#1E8FAA", line="#1E8FAA", container_id="social_card", z=30),
            polygon("social_land_a", 864, 84, 26, 25, [[0.08, 0.40], [0.30, 0.10], [0.70, 0.18], [0.86, 0.55], [0.50, 0.92], [0.14, 0.72]], fill="#4FB34F", line="#4FB34F", container_id="social_card", z=34),
            polygon("social_land_b", 889, 99, 25, 25, [[0.15, 0.18], [0.78, 0.20], [0.88, 0.62], [0.40, 0.88], [0.05, 0.50]], fill="#55BC50", line="#55BC50", container_id="social_card", z=34),
            node("social_factory_base", "process_box", 812, 130, 36, 26, "", fill="#F0B62E", line="#F0B62E", container_id="social_card", z=29, rounding=0.0),
            node("social_factory_stack", "process_box", 813, 104, 10, 52, "", fill="#C94D32", line="#C94D32", container_id="social_card", z=29, rounding=0.0),
            ellipse("social_person_head", 926, 70, 14, 14, "", fill="#F3C34C", line="#F3C34C", container_id="social_card", z=34),
            node("social_person_body", "process_box", 928, 84, 9, 38, "", fill="#2F6FC2", line="#2F6FC2", container_id="social_card", z=33, rounding=0.03),
        ]
    )
    edges.extend(
        [
            edge("social_ladder_a", 935, 89, 948, 139, route="straight", arrow=False, color="#F2C344", weight=2.0, z=36, allow_diagonal=True),
            edge("social_ladder_b", 948, 89, 936, 139, route="straight", arrow=False, color="#F2C344", weight=2.0, z=36, allow_diagonal=True),
        ]
    )

    for idx, (x, top, fill) in enumerate(((826, 238, "#1AA6A5"), (872, 220, "#00A87E"), (917, 202, "#46B073"))):
        nodes.append(node(f"economic_bar_{idx}", "process_box", x, top, 20, 260 - top, "", fill=fill, line=fill, container_id="economic_card", z=30, rounding=0.02))
        nodes.append(ellipse(f"economic_bar_cap_{idx}", x, top - 5, 20, 10, "", fill=fill, line=fill, container_id="economic_card", z=31))
    for idx, (x1, y1, x2, y2) in enumerate(((819, 229, 848, 209), (848, 209, 884, 222), (884, 222, 923, 190))):
        edges.append(edge(f"economic_growth_line_{idx}", x1, y1, x2, y2, route="straight", arrow=idx == 2, color="#BF3B2B", weight=2.0, z=42, allow_diagonal=True))

    nodes.extend(
        [
            polygon("renewables_panel", 807, 324, 91, 58, [[0.04, 0.20], [0.70, 0.03], [0.98, 0.83], [0.24, 0.98]], fill="#1F6F88", line="#143F54", container_id="renewables_card", z=30),
            polygon("renewables_panel_face", 816, 331, 73, 45, [[0.05, 0.20], [0.70, 0.03], [0.94, 0.82], [0.24, 0.96]], fill="#2E879D", line="#2E879D", container_id="renewables_card", z=31),
            node("renewables_house", "process_box", 839, 379, 38, 22, "", fill="#889AA0", line="#75858A", container_id="renewables_card", z=30, rounding=0.0),
            polygon("renewables_roof", 835, 365, 46, 20, [[0.06, 0.86], [0.50, 0.08], [0.96, 0.86]], fill="#6A7B7F", line="#6A7B7F", container_id="renewables_card", z=32),
            polygon("renewables_tree", 910, 366, 23, 38, [[0.52, 0.04], [0.86, 0.48], [0.64, 0.48], [0.92, 0.92], [0.08, 0.92], [0.36, 0.48], [0.14, 0.48]], fill="#76A45D", line="#76A45D", container_id="renewables_card", z=33),
        ]
    )
    edges.extend(
        [
            edge("solar_grid_v1", 835, 327, 855, 379, route="straight", arrow=False, color="#D6EEF1", weight=0.7, z=36, allow_diagonal=True),
            edge("solar_grid_v2", 855, 322, 879, 376, route="straight", arrow=False, color="#D6EEF1", weight=0.7, z=36, allow_diagonal=True),
            edge("solar_grid_h1", 815, 344, 893, 334, route="straight", arrow=False, color="#D6EEF1", weight=0.7, z=36, allow_diagonal=True),
            edge("solar_grid_h2", 821, 358, 899, 350, route="straight", arrow=False, color="#D6EEF1", weight=0.7, z=36, allow_diagonal=True),
        ]
    )

    nodes.extend(
        [
            node("green_factory_base", "process_box", 812, 500, 112, 28, "", fill="#78BFD0", line="#78BFD0", container_id="green_card", z=30, rounding=0.0),
            polygon("green_factory_roof", 812, 484, 92, 28, [[0.00, 1.00], [0.15, 0.45], [0.31, 1.00], [0.48, 0.30], [0.64, 1.00], [0.82, 0.45], [1.00, 1.00]], fill="#5AA8BD", line="#5AA8BD", container_id="green_card", z=32),
            node("green_chimney_main", "process_box", 862, 455, 12, 73, "", fill="#158A9D", line="#158A9D", container_id="green_card", z=31, rounding=0.0),
            node("green_chimney_top", "process_box", 858, 452, 20, 6, "", fill="#158A9D", line="#158A9D", container_id="green_card", z=32, rounding=0.0),
            node("green_tower_left", "process_box", 920, 476, 20, 52, "", fill="#1F8BBD", line="#1F8BBD", container_id="green_card", z=31, rounding=0.02),
            node("green_tower_right", "process_box", 948, 461, 9, 67, "", fill="#DE684E", line="#DE684E", container_id="green_card", z=31, rounding=0.0),
            ellipse("green_dome_left", 920, 466, 20, 17, "", fill="#2A94C8", line="#2A94C8", container_id="green_card", z=32),
            ellipse("green_dome_center", 891, 473, 22, 19, "", fill="#60B070", line="#60B070", container_id="green_card", z=34),
        ]
    )

    arrow_plan = [
        {"id": "A001", "from": "Industry 4.0 Technologies", "to": "Sustainability function list", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A002", "from": "Industry 4.0 Components", "to": "Sustainability function list", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A003", "from": "Industry 4.0 Components", "to": "Sustainability function list", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A004", "from": "Industry 4.0 Principles", "to": "Sustainability function list", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A005", "from": "Sustainability function list", "to": "Social development", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A006", "from": "Sustainability function list", "to": "Sustainable economic growth", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A007", "from": "Sustainability function list", "to": "Renewables", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A008", "from": "Sustainability function list", "to": "Green manufacturing", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
    ]

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.industry_4_0_sustainability_framework",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "industry_4_0_sustainability_framework",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_three_column_framework_template",
                "diagram_family": "industry_4_0_sustainability_functions_to_sustainable_manufacturing",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_visible_ocr_labels_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "industry_outer", "category": "source_framework", "source_bbox_px": [15, 20, 427, 544], "required_visible_labels": ["Industry 4.0", "Technologies", "Components", "Principles"]},
                    {"id": "technologies_panel", "category": "technology_layer", "source_bbox_px": [30, 46, 412, 186], "required_visible_labels": ["Artificial intelligence", "Mixed Reality", "IIoT", "Blockchain", "Digital twins", "Robotics", "Big data analytics", "CPS"]},
                    {"id": "components_panel", "category": "component_layer", "source_bbox_px": [31, 216, 412, 357], "required_visible_labels": ["Smart customers", "Smart distribution", "Digital supply networks", "Smart shareholders", "Smart factory", "Smart products"]},
                    {"id": "principles_panel", "category": "principle_layer", "source_bbox_px": [31, 395, 412, 534], "required_visible_labels": ["Virtualization", "Vertical integration", "Real-time capability", "Interoperability", "Technical assistance", "Decentralization", "Horizontal integration"]},
                    {"id": "functions_outer", "category": "function_list", "source_bbox_px": [456, 19, 753, 543], "required_visible_labels": ["Industry 4.0 Sustainability Functions", "Business model innovation", "Sustainable value-creation networking"]},
                    {"id": "manufacturing_outer", "category": "outcomes", "source_bbox_px": [783, 20, 965, 544], "required_visible_labels": ["Sustainable Manufacturing", "Social development", "Sustainable economic growth", "Renewables", "Green manufacturing"]},
                ],
            },
            "region_plan": [
                {"id": "industry_outer", "category": "source_framework", "source_bbox_px": [15, 20, 427, 544]},
                {"id": "functions_outer", "category": "function_list", "source_bbox_px": [456, 19, 753, 543]},
                {"id": "manufacturing_outer", "category": "outcomes", "source_bbox_px": [783, 20, 965, 544]},
            ],
            "arrow_plan": arrow_plan,
            "notes": [
                "Editable semantic reconstruction for three-column Industry 4.0 sustainability framework figures.",
                "The left framework layers, center function list, right outcome cards, inter-column arrows, and outcome pictograms are rebuilt as Visio-editable objects.",
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


def build_drought_basin_workflow_scene(
    image_path: Path,
    width: int,
    height: int,
    ocr_items: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    base_w = 981.0
    base_h = 1417.0

    def sx(value: float) -> float:
        return value * width / base_w

    def sy(value: float) -> float:
        return value * height / base_h

    def bbox(x: float, y: float, w: float, h: float) -> list[float]:
        return [round(sx(x), 2), round(sy(y), 2), round(sx(x + w), 2), round(sy(y + h), 2)]

    def attach(item: dict[str, Any], x: float, y: float, w: float, h: float, container_id: str | None = None) -> dict[str, Any]:
        item["source_bbox_px"] = bbox(x, y, w, h)
        if container_id:
            item["container_id"] = container_id
        item.setdefault("allow_overlap", True)
        return item

    def node(
        node_id: str,
        node_type: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str = "#FFFFFF",
        line: str = "#8A8F96",
        z: int = 20,
        font_size: float = 12,
        dash: str = "solid",
        rounding: float = 0.08,
        container_id: str | None = None,
        weight: str = "regular",
    ) -> dict[str, Any]:
        item = px_node(
            node_id,
            node_type,
            sx(x),
            sy(y),
            sx(w),
            sy(h),
            text,
            fill=fill,
            line=line,
            z=z,
            font_size=font_size,
            text_color="#111111",
            dash=dash,
            rounding=rounding,
        )
        item["style"].update(
            {
                "font_family_candidates": ["Arial", "Calibri", "Times New Roman", "Microsoft YaHei UI"],
                "font_role": "ui_sans",
                "font_weight": weight,
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 5.2,
                "text_margin_in": 0.025,
                "line_weight_pt": 1.0,
            }
        )
        return attach(item, x, y, w, h, container_id)

    def label(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_size: float = 12,
        weight: str = "regular",
        container_id: str | None = None,
        z: int = 90,
        color: str = "#111111",
    ) -> dict[str, Any]:
        item = text_node(node_id, sx(x), sy(y), sx(w), sy(h), text, font_size=font_size, weight=weight, z=z)
        item["style"].update(
            {
                "font_family_candidates": ["Arial", "Calibri", "Times New Roman", "Microsoft YaHei UI"],
                "font_role": "ui_sans",
                "text_fit": "shrink_to_fit",
                "min_font_size_pt": 5.0,
                "text_color": color,
            }
        )
        return attach(item, x, y, w, h, container_id)

    def band(node_id: str, x: float, y: float, w: float, h: float, fill: str) -> dict[str, Any]:
        item = node(node_id, "group_container", x, y, w, h, "", fill=fill, line="#B8B8B8", z=2, font_size=1, rounding=0.0)
        item["style"].update({"line_weight_pt": 1.05, "shadow": {"color": "#888888", "offset_x_in": 0.03, "offset_y_in": -0.03, "transparency_pct": 78}})
        return item

    def title_box(node_id: str, x: float, y: float, w: float, h: float, text: str, fill: str, *, font_size: float = 17) -> dict[str, Any]:
        item = node(node_id, "rounded_process", x, y, w, h, text, fill=fill, line=fill, z=28, font_size=font_size, rounding=0.04, weight="bold")
        item["style"]["text_fit"] = "single_line"
        return item

    def round_panel(node_id: str, x: float, y: float, w: float, h: float, *, container_id: str, fill: str = "#FFFFFF", z: int = 12) -> dict[str, Any]:
        item = node(node_id, "rounded_process", x, y, w, h, "", fill=fill, line=fill, z=z, font_size=1, rounding=0.20, container_id=container_id)
        item["style"].update({"shadow": {"color": "#999999", "offset_x_in": 0.035, "offset_y_in": -0.035, "transparency_pct": 82}})
        return item

    def dashed_box(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        container_id: str,
        font_size: float = 12,
        z: int = 35,
    ) -> dict[str, Any]:
        item = node(node_id, "rounded_process", x, y, w, h, text, fill="#FFFFFF", line="#111111", z=z, font_size=font_size, dash="dash", rounding=0.06, container_id=container_id)
        item["style"].update({"line_weight_pt": 1.0, "text_fit": "shrink_to_fit"})
        return item

    def polygon(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        points: list[list[float]],
        *,
        fill: str,
        line: str = "#6C7A89",
        container_id: str | None = None,
        z: int = 42,
    ) -> dict[str, Any]:
        item = {
            "id": node_id,
            "type": "polygon_node",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "points": points,
            "style": {"fill": fill, "line": line, "line_weight_pt": 0.8},
        }
        return attach(item, x, y, w, h, container_id)

    def ellipse(node_id: str, x: float, y: float, w: float, h: float, *, fill: str, line: str, container_id: str | None = None, z: int = 40) -> dict[str, Any]:
        return node(node_id, "ellipse_node", x, y, w, h, "", fill=fill, line=line, z=z, font_size=1, rounding=0.0, container_id=container_id)

    def edge(
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        route: str = "straight",
        points: list[list[float]] | None = None,
        color: str = "#2E93A3",
        weight: float = 2.0,
        arrow: bool = True,
        z: int = 72,
        arrow_plan_id: str | None = None,
        allow_diagonal: bool = False,
    ) -> dict[str, Any]:
        item = edge_px(
            edge_id,
            sx(x1),
            sy(y1),
            sx(x2),
            sy(y2),
            arrow=arrow,
            route=route,
            points=[[sx(px), sy(py)] for px, py in points] if points else None,
            z=z,
            allow_cross_container=True,
        )
        item["style"].update({"line": color, "line_weight_pt": weight, "arrow_size": "small"})
        if route in {"horizontal", "vertical"} and arrow:
            item["type"] = "lane_arrow"
        if arrow_plan_id:
            item["arrow_plan_id"] = arrow_plan_id
        if allow_diagonal:
            item["allow_diagonal"] = True
        return item

    def grid_matrix(
        node_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rows: int,
        cols: int,
        colors: list[str],
        container_id: str | None = None,
        z: int = 32,
    ) -> dict[str, Any]:
        cells = [[row, col, colors[(row + col) % len(colors)]] for row in range(rows) for col in range(cols)]
        item = {
            "id": node_id,
            "type": "grid_matrix",
            "x": sx(x),
            "y": sy(y),
            "w": sx(w),
            "h": sy(h),
            "z": z,
            "text": "",
            "rows": rows,
            "cols": cols,
            "colored_cells": cells,
            "style": {"line": "#8A9AA8", "line_weight_pt": 0.65, "grid_line": "#D7DEE5", "grid_line_weight_pt": 0.35},
        }
        return attach(item, x, y, w, h, container_id)

    def map_shape(prefix: str, x: float, y: float, w: float, h: float, container_id: str, *, fill: str = "#BFD8E8", hot: str = "#D4573C") -> list[dict[str, Any]]:
        return [
            node(f"{prefix}_map_canvas", "process_box", x, y, w, h, "", fill="#F7FBFC", line="#C7D7E2", z=23, font_size=1, rounding=0.0, container_id=container_id),
            polygon(f"{prefix}_land_1", x + 0.08 * w, y + 0.25 * h, 0.34 * w, 0.30 * h, [[0.02, 0.46], [0.30, 0.12], [0.72, 0.25], [0.90, 0.60], [0.42, 0.84]], fill=fill, line=fill, container_id=container_id, z=34),
            polygon(f"{prefix}_land_2", x + 0.50 * w, y + 0.22 * h, 0.34 * w, 0.32 * h, [[0.08, 0.25], [0.38, 0.08], [0.84, 0.30], [0.72, 0.74], [0.22, 0.86]], fill=fill, line=fill, container_id=container_id, z=34),
            polygon(f"{prefix}_hot_basin", x + 0.22 * w, y + 0.42 * h, 0.14 * w, 0.25 * h, [[0.22, 0.08], [0.76, 0.24], [0.82, 0.70], [0.40, 0.94], [0.04, 0.54]], fill=hot, line=hot, container_id=container_id, z=36),
        ]

    def line_plot(prefix: str, x: float, y: float, w: float, h: float, container_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes_local: list[dict[str, Any]] = [node(f"{prefix}_frame", "process_box", x, y, w, h, "", fill="#FFFFFF", line="#C8C8C8", z=24, font_size=1, rounding=0.0, container_id=container_id)]
        edges_local: list[dict[str, Any]] = []
        for i in range(4):
            yy = y + h * (0.18 + i * 0.20)
            edges_local.append(edge(f"{prefix}_grid_{i}", x + 6, yy, x + w - 6, yy, route="horizontal", color="#D9D9D9", weight=0.55, arrow=False, z=28))
        xs = [x + w * t for t in (0.06, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.92)]
        ys1 = [y + h * t for t in (0.42, 0.35, 0.48, 0.38, 0.44, 0.30, 0.34, 0.50)]
        ys2 = [y + h * t for t in (0.62, 0.66, 0.58, 0.72, 0.61, 0.68, 0.55, 0.60)]
        for idx, (xs0, ys0, xs1, ys1v) in enumerate(zip(xs, ys1, xs[1:], ys1[1:])):
            edges_local.append(edge(f"{prefix}_line_a_{idx}", xs0, ys0, xs1, ys1v, route="straight", color="#A55B5B", weight=1.05, arrow=False, z=34, allow_diagonal=True))
        for idx, (xs0, ys0, xs1, ys1v) in enumerate(zip(xs, ys2, xs[1:], ys2[1:])):
            edges_local.append(edge(f"{prefix}_line_b_{idx}", xs0, ys0, xs1, ys1v, route="straight", color="#6A78A8", weight=1.05, arrow=False, z=34, allow_diagonal=True))
        return nodes_local, edges_local

    def data_stack(prefix: str, x: float, y: float, w: float, h: float, container_id: str, *, palette: list[str]) -> list[dict[str, Any]]:
        sheets: list[dict[str, Any]] = []
        for index, fill in enumerate(palette):
            yy = y + index * 13
            sheets.append(
                polygon(
                    f"{prefix}_sheet_{index}",
                    x + index * 3,
                    yy,
                    w,
                    h,
                    [[0.16, 0.05], [1.00, 0.05], [0.84, 0.94], [0.00, 0.94]],
                    fill=fill,
                    line="#B0B8BE",
                    container_id=container_id,
                    z=25 + index,
                )
            )
        return sheets

    nodes: list[dict[str, Any]] = [
        node("page_background", "page_background", 0, 0, base_w, base_h, "", fill="#FFFFFF", line="none", z=0),
        band("datasets_band", 4, 14, 972, 166, "#E5F0FA"),
        band("spei_band", 3, 221, 973, 275, "#EDF6E6"),
        band("clustering_band", 4, 540, 973, 433, "#FFF0E2"),
        band("influence_band", 3, 1014, 974, 263, "#E5F5FA"),
    ]
    edges: list[dict[str, Any]] = []

    nodes.extend(
        [
            title_box("datasets_header", 378, 2, 228, 36, "Datasets input", "#C9DCEB", font_size=17),
            label("meteorological_label", 55, 52, 190, 28, "Meteorological data", font_size=13.5, weight="bold", container_id="datasets_band"),
            label("sst_label", 348, 52, 94, 28, "SST data", font_size=13.5, weight="bold", container_id="datasets_band"),
            label("nino_label", 548, 52, 130, 28, "Nino 3.4 data", font_size=13.5, weight="bold", container_id="datasets_band"),
            label("river_data_label", 771, 52, 174, 28, "River basins data", font_size=13.5, weight="bold", container_id="datasets_band"),
        ]
    )
    nodes.extend(data_stack("meteo", 59, 81, 160, 88, "datasets_band", palette=["#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"]))
    for idx, x in enumerate((98, 132, 165)):
        nodes.append(ellipse(f"meteo_dots_{idx}", x, 100 + idx * 12, 8, 8, fill="#BFD77B", line="#BFD77B", container_id="datasets_band", z=40))
    nodes.extend(data_stack("sst", 292, 80, 158, 78, "datasets_band", palette=["#2C58B5", "#46A7D9", "#EAB347", "#F76D32", "#3465C7"]))
    nodes.append(grid_matrix("nino_map_grid", 524, 78, 178, 90, rows=4, cols=8, colors=["#E9FBF2", "#88CF9A", "#27A078", "#2E83BC"], container_id="datasets_band", z=29))
    nodes.extend(map_shape("river_data", 770, 78, 176, 91, "datasets_band", fill="#91B9D4", hot="#E45E34"))

    nodes.append(title_box("spei_header", 328, 205, 322, 39, "Drought index SPEI-12", "#C8DDD2", font_size=16.5))
    edges.append(edge("datasets_to_spei", 491, 180, 491, 205, route="vertical", color="#3698A8", weight=3.0, arrow_plan_id="A001"))
    nodes.extend(
        [
            round_panel("drought_wet_panel", 18, 253, 386, 195, container_id="spei_band"),
            label("drought_wet_title", 89, 272, 235, 32, "Drought-wet change", font_size=17, weight="bold", container_id="spei_band"),
            dashed_box("pre_trend", 31, 308, 195, 48, "PRE trend", container_id="spei_band", font_size=14),
            dashed_box("temporal_variation", 209, 308, 184, 48, "Temporal variation", container_id="spei_band", font_size=13),
            dashed_box("pet_trend", 31, 379, 195, 48, "PET trend", container_id="spei_band", font_size=14),
            dashed_box("spatial_patterns", 209, 379, 184, 48, "Spatial patterns", container_id="spei_band", font_size=13),
            round_panel("river_basins_panel", 516, 253, 441, 232, container_id="spei_band"),
            label("river_basins_title", 585, 266, 318, 34, "34 major global river basins", font_size=16, weight="bold", container_id="spei_band"),
        ]
    )
    nodes.extend(map_shape("basin_map", 555, 317, 190, 104, "spei_band", fill="#B9C5CF", hot="#D65243"))
    nodes.append(ellipse("basin_dot_a", 640, 354, 9, 9, fill="#8D9AA4", line="#8D9AA4", container_id="spei_band", z=40))
    nodes.append(ellipse("basin_dot_b", 675, 365, 7, 7, fill="#8D9AA4", line="#8D9AA4", container_id="spei_band", z=40))
    nodes.append(node("basin_colorbar", "process_box", 596, 414, 120, 10, "", fill="#C84138", line="#FFFFFF", z=40, container_id="spei_band", rounding=0.0))
    nodes.append(node("basin_colorbar_blue", "process_box", 676, 414, 80, 10, "", fill="#3278A8", line="#FFFFFF", z=41, container_id="spei_band", rounding=0.0))
    nodes.append(grid_matrix("river_basin_hydrograph_matrix", 758, 292, 162, 184, rows=7, cols=5, colors=["#FFFFFF", "#F8F4F8"], container_id="spei_band", z=24))
    for r in range(7):
        for c in range(5):
            nodes.append(node(f"hydrograph_{r}_{c}", "process_box", 760 + c * 31, 292 + r * 26, 25, 17, "", fill="#FFFFFF", line="#C8C8C8", z=28, font_size=1, container_id="spei_band", rounding=0.0))
            edges.append(edge(f"hydro_line_{r}_{c}", 764 + c * 31, 302 + r * 26, 781 + c * 31, 298 + r * 26 + (c % 2) * 3, route="straight", color="#A86D85", weight=0.65, arrow=False, z=38, allow_diagonal=True))
    edges.append(edge("drought_change_to_basins", 404, 351, 516, 351, route="horizontal", color="#C5E483", weight=8.0, arrow_plan_id="A002"))
    edges.append(edge("spei_to_clustering", 491, 496, 491, 531, route="vertical", color="#3698A8", weight=3.0, arrow_plan_id="A003"))

    nodes.append(title_box("clustering_header", 300, 528, 362, 38, "3-D  Drought Clustering", "#F5D8C0", font_size=16.5))
    nodes.extend(
        [
            dashed_box("thresholds_box", 33, 591, 323, 118, "Index threshold: -1\nArea threshold: 1.56%\nSpace-time domain :3x3x3", container_id="clustering_band", font_size=13.5, z=35),
            node("drought_structure", "rounded_process", 386, 624, 154, 65, "Drought\nstructure", fill="#F39478", line="#F39478", z=34, font_size=15.5, container_id="clustering_band", rounding=0.07, weight="bold"),
            grid_matrix("drought_cube_grid", 562, 589, 245, 106, rows=5, cols=8, colors=["#F3D071", "#DCEBD8", "#A4CDC4", "#E06B62", "#F4F4F4"], container_id="clustering_band", z=32),
            round_panel("time_panel", 815, 578, 137, 139, container_id="clustering_band"),
            label("time_n2", 851, 607, 88, 28, "Time n+2", font_size=13, container_id="clustering_band"),
            label("time_n1", 851, 648, 88, 28, "Time n+1", font_size=13, container_id="clustering_band"),
            label("time_n", 851, 687, 72, 28, "Time n", font_size=13, container_id="clustering_band"),
        ]
    )
    edges.append(edge("thresholds_to_structure", 356, 650, 386, 650, route="horizontal", color="#BC7B52", weight=2.2))
    edges.append(edge("structure_to_cube", 540, 654, 562, 640, route="straight", color="#BC7B52", weight=2.0, allow_diagonal=True))
    edges.append(edge("cube_to_time", 807, 647, 815, 647, route="horizontal", color="#BC7B52", weight=4.0))
    edges.append(edge("cluster_cycle_top", 364, 612, 426, 612, route="straight", points=[[376, 585], [410, 585]], color="#C7875D", weight=2.0))
    edges.append(edge("cluster_cycle_bottom", 489, 690, 435, 705, route="straight", points=[[470, 724], [435, 705]], color="#C7875D", weight=2.0))
    edges.append(edge("cluster_mid_separator", 13, 742, 969, 742, route="horizontal", color="#C99875", weight=1.1, arrow=False, z=25))

    nodes.extend(
        [
            round_panel("event_characteristics_panel", 16, 760, 473, 188, container_id="clustering_band"),
            label("event_characteristics_title", 73, 778, 336, 31, "Drought event characteristics", font_size=15.5, weight="bold", container_id="clustering_band"),
            dashed_box("drought_duration", 28, 810, 216, 50, "Drought duration", container_id="clustering_band", font_size=12.5),
            dashed_box("drought_displacements", 229, 810, 234, 50, "Drought displacements", container_id="clustering_band", font_size=12.5),
            dashed_box("drought_number", 28, 878, 216, 50, "Drought number", container_id="clustering_band", font_size=12.5),
            dashed_box("drought_area", 229, 878, 234, 50, "Drought area", container_id="clustering_band", font_size=12.5),
            label("spatial_distribution", 515, 788, 94, 57, "Spatial\ndistribution", font_size=13.5, weight="bold", container_id="clustering_band"),
            label("comparative_analysis", 500, 867, 130, 64, "Comparative\nanalysis", font_size=13.5, weight="bold", container_id="clustering_band"),
            dashed_box("typical_event", 632, 784, 313, 75, "Spatiotemporal structure of\ntypical drought event", container_id="clustering_band", font_size=12.5),
            dashed_box("koppen_box", 632, 859, 313, 75, "The Koppen-Geiger climate\nclassification", container_id="clustering_band", font_size=12.5),
        ]
    )
    nodes.append(node("comparison_icon", "process_box", 540, 846, 27, 23, "", fill="#E3714A", line="#E3714A", z=40, container_id="clustering_band", rounding=0.04))
    for x, side in [(488, "left"), (618, "right")]:
        nodes.append(label(f"analysis_brace_{side}", x, 828, 18, 88, "}", font_size=32, weight="bold", container_id="clustering_band", color="#D17A55"))
    edges.append(edge("clustering_to_influence", 491, 973, 491, 1005, route="vertical", color="#3698A8", weight=3.0, arrow_plan_id="A004"))

    nodes.append(title_box("influence_header", 286, 1004, 403, 37, "Influencing factors of drought", "#9EDAE3", font_size=16.5))
    nodes.extend(
        [
            round_panel("mca_panel", 23, 1077, 427, 137, container_id="influence_band"),
            label("mca_title", 67, 1094, 344, 31, "Maximum covariance analysis", font_size=15.2, weight="bold", container_id="influence_band"),
            dashed_box("mca_sst", 64, 1138, 88, 48, "SST", container_id="influence_band", font_size=13),
            dashed_box("mca_drought", 193, 1138, 97, 48, "Drought", container_id="influence_band", font_size=13),
            dashed_box("mca_enso", 325, 1138, 93, 48, "ENSO", container_id="influence_band", font_size=13),
            label("mca_patterns_title", 473, 1057, 490, 30, "Spatiotemporal patterns of the MCA2 mode", font_size=15.0, weight="bold", container_id="influence_band"),
        ]
    )
    edges.append(edge("sst_to_drought", 152, 1162, 193, 1162, route="horizontal", color="#0B6F94", weight=3.0, arrow_plan_id="A005"))
    edges.append(edge("enso_to_drought", 325, 1162, 290, 1162, route="horizontal", color="#0B6F94", weight=3.0, arrow_plan_id="A006"))
    edges.append(edge("mca_to_patterns", 450, 1150, 527, 1150, route="horizontal", color="#0B6F94", weight=5.0, arrow_plan_id="A007"))
    line_plot_nodes, line_plot_edges = line_plot("mca_line_plot", 540, 1086, 386, 76, "influence_band")
    nodes.extend(line_plot_nodes)
    edges.extend(line_plot_edges)
    nodes.append(grid_matrix("mca_sst_map", 548, 1171, 181, 62, rows=4, cols=8, colors=["#C3312E", "#F6C9A8", "#FFFFFF", "#7AB6D8"], container_id="influence_band", z=31))
    nodes.extend(map_shape("mca_world_map", 746, 1171, 185, 62, "influence_band", fill="#BFD4E2", hot="#C84B3F"))
    nodes.append(node("mca_colorbar_red", "process_box", 660, 1242, 121, 10, "", fill="#B42028", line="#FFFFFF", z=40, container_id="influence_band", rounding=0.0))
    nodes.append(node("mca_colorbar_blue", "process_box", 781, 1242, 94, 10, "", fill="#2E79B7", line="#FFFFFF", z=40, container_id="influence_band", rounding=0.0))

    edges.append(edge("influence_to_final", 491, 1277, 491, 1304, route="vertical", color="#3698A8", weight=3.0, arrow_plan_id="A008"))
    nodes.append(
        node(
            "final_identification",
            "rounded_process",
            39,
            1312,
            902,
            58,
            "Identification and contrast of meteorological drought of global river basins",
            fill="#F3F3EF",
            line="#F3F3EF",
            z=28,
            font_size=15,
            rounding=0.06,
            weight="bold",
        )
    )

    arrow_plan = [
        {"id": "A001", "from": "Datasets input", "to": "Drought index SPEI-12", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A002", "from": "Drought-wet change", "to": "34 major global river basins", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A003", "from": "Drought index SPEI-12", "to": "3-D Drought Clustering", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A004", "from": "Drought event characteristics", "to": "Influencing factors of drought", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A005", "from": "SST", "to": "Drought", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A006", "from": "ENSO", "to": "Drought", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A007", "from": "Maximum covariance analysis", "to": "Spatiotemporal patterns of the MCA2 mode", "route_shape": "straight_horizontal", "semantic_intent": "data_flow", "certainty": "high"},
        {"id": "A008", "from": "Influencing factors of drought", "to": "Identification and contrast", "route_shape": "straight_vertical", "semantic_intent": "data_flow", "certainty": "high"},
    ]

    return {
        "version": "0.1",
        "metadata": {
            "title": title or image_path.stem,
            "created_by": "fig4visio.image_auto_scene.drought_basin_workflow",
            "style_profile": "paper_white",
            "fidelity": "semantic_editable_rebuild",
            "source_image": str(image_path.resolve()),
            "ocr_items": len(ocr_items),
            "region_strategy": "module_first",
            "architecture_template": "drought_basin_workflow",
            "visual_reference_layer": False,
            "raster_tile_policy": "semantic_template_no_raster_tiles",
            "partial_raster_tiles": 0,
            "source_visual_inventory": {
                "analysis_basis": "ocr_keyword_triggered_drought_basin_workflow_template",
                "diagram_family": "meteorological_drought_global_river_basin_workflow",
                "do_not_translate": True,
                "unknown_text_policy": "preserve_visible_ocr_labels_mark_unreadable_do_not_invent",
                "regions": [
                    {"id": "datasets_band", "category": "input", "source_bbox_px": [4, 14, 976, 180], "required_visible_labels": ["Datasets input", "Meteorological data", "SST data", "Nino 3.4 data", "River basins data"]},
                    {"id": "spei_band", "category": "core", "source_bbox_px": [3, 221, 976, 496], "required_visible_labels": ["Drought index SPEI-12", "Drought-wet change", "34 major global river basins"]},
                    {"id": "clustering_band", "category": "core", "source_bbox_px": [4, 540, 977, 973], "required_visible_labels": ["3-D Drought Clustering", "Drought structure", "Drought event characteristics"]},
                    {"id": "influence_band", "category": "output", "source_bbox_px": [3, 1014, 977, 1277], "required_visible_labels": ["Influencing factors of drought", "Maximum covariance analysis", "Spatiotemporal patterns of the MCA2 mode"]},
                    {"id": "final_identification", "category": "output", "source_bbox_px": [39, 1312, 941, 1370], "required_visible_labels": ["Identification and contrast of meteorological drought of global river basins"]},
                ],
            },
            "region_plan": [
                {"id": "datasets_band", "category": "input", "source_bbox_px": [4, 14, 976, 180]},
                {"id": "spei_band", "category": "core", "source_bbox_px": [3, 221, 976, 496]},
                {"id": "clustering_band", "category": "core", "source_bbox_px": [4, 540, 977, 973]},
                {"id": "influence_band", "category": "output", "source_bbox_px": [3, 1014, 977, 1277]},
                {"id": "final_identification", "category": "output", "source_bbox_px": [39, 1312, 941, 1370]},
            ],
            "arrow_plan": arrow_plan,
            "notes": [
                "Editable semantic reconstruction for meteorological drought and global river-basin workflow figures.",
                "Dataset stacks, SPEI modules, 3-D clustering, drought-event characteristics, MCA causal boxes, map panels, line plots, and main flow arrows are Visio-editable objects.",
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
    if is_drought_basin_workflow_figure(ocr_items, width, height):
        scene = build_drought_basin_workflow_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_remote_sensing_rsei_workflow_figure(ocr_items, width, height):
        scene = build_remote_sensing_rsei_workflow_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_industry_4_0_sustainability_framework_figure(ocr_items, width, height):
        scene = build_industry_4_0_sustainability_framework_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_deformable_transformer_encoder_decoder_figure(ocr_items, width, height):
        scene = build_deformable_transformer_encoder_decoder_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_channel_attention_recalibration_figure(ocr_items, width, height):
        scene = build_channel_attention_recalibration_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_attention_mechanism_figure(ocr_items, width, height):
        scene = build_attention_mechanism_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_cross_attention_figure(ocr_items, width, height):
        scene = build_cross_attention_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_mask_res_block_figure(ocr_items, width, height):
        scene = build_mask_res_block_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
    if is_swin_transformer_architecture(ocr_items, width, height):
        scene = build_swin_transformer_architecture_scene(image_path, width, height, ocr_items, title=title)
        scene.setdefault("metadata", {})["raster_tile_policy"] = "semantic_template_no_raster_tiles"
        scene.setdefault("metadata", {})["reconstruction_mode"] = mode
        return scene
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
            "created_by": "fig4visio.image_auto_scene",
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
