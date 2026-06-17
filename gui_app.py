# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk


APP_NAME = "Fig4Visio GUI"
MAX_PREVIEW_SIZE = (820, 560)
SELF_CHECK_THRESHOLD = 0.38
INITIAL_AUTO_ATTEMPTS = 3
MAX_AUTO_ATTEMPTS = 5
RECONSTRUCTION_MODE_SEQUENCE = [
    "standard",
    "vector_trace",
    "vector_trace_dense",
    "standard",
    "vector_trace_dense",
]


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent


APP_ROOT = runtime_root()
SCRIPTS_DIR = APP_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@dataclass
class OutputSet:
    run_dir: Path
    attempt_dir: Path
    source_image: Path
    source_manifest: Path
    scene: Path
    output_dir: Path
    basename: str
    vsdx: Path
    png: Path
    svg: Path
    review_dir: Path
    review_manifest: Path
    quality_json: Path
    quality_md: Path
    complexity_report: Path
    audit_report: Path
    pair_preview: Path | None
    self_check_json: Path
    self_check_png: Path
    self_check_passed: bool
    self_check_score: float
    download_allowed: bool
    quality_status: str
    quality_label: str
    quality_summary: str


def safe_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".scene"):
        stem = stem[: -len(".scene")]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return value or "fig4visio_output"


def load_script_module(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_source_image(source_path: Path, run_dir: Path, figure_id: str) -> tuple[Path, Path]:
    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix.lower() or ".png"
    staged_path = source_dir / f"original{suffix}"
    shutil.copy2(source_path, staged_path)
    manifest_path = source_dir / "source_manifest.json"
    manifest = {
        "figure_id": figure_id,
        "canonical_source_image": str(staged_path.resolve()),
        "canonical_source_sha256": sha256_file(staged_path),
        "original_input_path": str(source_path.resolve()),
        "original_input_sha256": sha256_file(source_path),
        "source_dir": str(source_dir.resolve()),
        "workflow_note": "GUI workflow stages a stable local source before reconstruction and review.",
    }
    write_json(manifest_path, manifest)
    return staged_path, manifest_path


def auto_scene_for_image(image_path: Path, scene_path: Path, *, reconstruction_mode: str) -> Path:
    image_auto_scene = load_script_module("image_auto_scene")
    scene = image_auto_scene.build_scene(
        image_path,
        title=image_path.stem,
        allow_raster_tiles=False,
        reconstruction_mode=reconstruction_mode,
    )
    metadata = scene.setdefault("metadata", {})
    metadata["gui_workflow"] = "self_check_retry_gate"
    metadata["source_embedding_policy"] = "no_full_source_image; GUI disables raster tiles by default"
    metadata["quality_claim"] = "editable_draft_requires_screenshot_self_check"
    metadata["gui_reconstruction_mode"] = reconstruction_mode
    notes = metadata.setdefault("notes", [])
    if isinstance(notes, list):
        notes.append("GUI workflow disables raster source tiles to avoid fake whole-image reconstruction.")
    write_json(scene_path, scene)
    return scene_path


def validate_scene_file(scene_path: Path, strict: bool) -> str:
    scene_validate = load_script_module("scene_validate")
    scene = read_json(scene_path)
    errors, warnings = scene_validate.validate_scene(scene, strict=strict)
    lines: list[str] = []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in errors)
        raise RuntimeError("\n".join(lines))
    lines.append(f"Scene is valid: {scene_path}")
    lines.append(
        f"Nodes: {len(scene.get('nodes', []))}, "
        f"Edges: {len(scene.get('edges', []))}, "
        f"Assets: {len(scene.get('assets', []))}"
    )
    return "\n".join(lines)


def render_scene(scene_path: Path, output_dir: Path, basename: str, *, visible: bool = False) -> str:
    scene_to_visio = load_script_module("scene_to_visio")
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "scene_to_visio.py",
        str(scene_path),
        "--output-dir",
        str(output_dir),
        "--basename",
        basename,
        "--skip-rebuild-gate",
    ]
    if visible:
        argv.append("--visible")

    buffer = io.StringIO()
    previous_argv = sys.argv[:]
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exit_code = scene_to_visio.main()
        if exit_code:
            raise RuntimeError(buffer.getvalue().strip() or f"scene_to_visio exited with {exit_code}")
    finally:
        sys.argv = previous_argv
    return buffer.getvalue().strip()


def write_scene_reports(scene_path: Path, report_dir: Path) -> tuple[Path, Path, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    complexity_path = report_dir / "scene_complexity.md"
    audit_path = report_dir / "scene_audit.md"
    log_lines: list[str] = []

    try:
        scene_complexity = load_script_module("scene_complexity")
        scene = scene_complexity.load_scene(scene_path)
        complexity_path.write_text(scene_complexity.scene_complexity_report(scene, strict=False) + "\n", encoding="utf-8")
        log_lines.append(f"已写入复杂度报告: {complexity_path}")
    except Exception as exc:
        complexity_path.write_text(f"Failed to generate complexity report:\n{exc}\n", encoding="utf-8")
        log_lines.append(f"复杂度报告生成失败，已记录: {exc}")

    try:
        scene_audit = load_script_module("scene_audit")
        scene = scene_audit.load_scene(scene_path)
        audit_path.write_text(scene_audit.audit_scene(scene), encoding="utf-8")
        log_lines.append(f"已写入模块审计报告: {audit_path}")
    except Exception as exc:
        audit_path.write_text(f"Failed to generate audit report:\n{exc}\n", encoding="utf-8")
        log_lines.append(f"模块审计报告生成失败，已记录: {exc}")

    return complexity_path, audit_path, "\n".join(log_lines)


def make_review_bundle(
    original: Path,
    replica: Path,
    scene_path: Path,
    review_dir: Path,
    basename: str,
) -> tuple[Path, Path | None, str]:
    make_review_assets = load_script_module("make_review_assets")
    review_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "make_review_assets.py",
        "--original",
        str(original),
        "--replica",
        str(replica),
        "--output-dir",
        str(review_dir),
        "--id",
        basename,
        "--round",
        "1",
        "--scene",
        str(scene_path),
        "--include-global-pair",
        "--crops",
        "left",
        "center",
        "right",
        "top",
        "middle",
        "bottom",
        "arrow_dense",
        "small_text",
        "--write-review-bundle",
    ]
    buffer = io.StringIO()
    previous_argv = sys.argv[:]
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exit_code = make_review_assets.main()
        if exit_code:
            raise RuntimeError(buffer.getvalue().strip() or f"make_review_assets exited with {exit_code}")
    finally:
        sys.argv = previous_argv
    manifest_path = review_dir / f"{basename}_review_manifest.json"
    pair_path = review_dir / f"{basename}_pair_global.png"
    return manifest_path, pair_path if pair_path.exists() else None, buffer.getvalue().strip()


def scene_quality_metrics(scene: dict, source_image: Path) -> dict[str, object]:
    page = scene.get("page", {}) if isinstance(scene.get("page"), dict) else {}
    page_width = float(page.get("width", 0) or 0)
    page_height = float(page.get("height", 0) or 0)
    page_area = max(1.0, page_width * page_height)
    aspect_ratio = page_width / page_height if page_height else 0.0
    nodes = [node for node in scene.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in scene.get("edges", []) if isinstance(edge, dict)]
    assets = [asset for asset in scene.get("assets", []) if isinstance(asset, dict)]
    metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}

    visible_nodes = [
        node
        for node in nodes
        if node.get("type") not in {"page_background", "audit_region"}
        and not str(node.get("id", "")).lower().endswith("background")
    ]
    image_nodes = [node for node in nodes if node.get("type") == "image_tile"]
    image_area = 0.0
    max_image_area = 0.0
    for node in image_nodes:
        try:
            area = float(node.get("w", 0) or 0) * float(node.get("h", 0) or 0)
        except (TypeError, ValueError):
            area = 0.0
        image_area += area
        max_image_area = max(max_image_area, area)

    source_hash = sha256_file(source_image)
    full_source_asset = False
    for asset in assets:
        asset_path = asset.get("path")
        if not asset_path:
            continue
        try:
            path = Path(str(asset_path))
            if path.exists() and sha256_file(path) == source_hash:
                full_source_asset = True
                break
        except Exception:
            continue

    reference_keys = ["source_reference_layer", "visual_reference_layer", "reference_layer", "hide_editable_overlay"]
    source_reference_node = any(
        any(token in str(node.get("id", "")).lower() for token in ("source_reference", "reference_layer", "original_image"))
        or any(token in str(node.get("type", "")).lower() for token in ("source_reference", "reference_layer"))
        for node in nodes
    )
    metadata_reference_layer = any(bool(metadata.get(key)) for key in reference_keys)
    text_nodes = [node for node in visible_nodes if str(node.get("text", node.get("symbol", ""))).strip()]
    max_asset_area_fraction = max_image_area / page_area
    total_asset_area_fraction = image_area / page_area
    blocked_embedding = (
        full_source_asset
        or source_reference_node
        or metadata_reference_layer
        or max_asset_area_fraction > 0.12
        or total_asset_area_fraction > 0.25
    )

    return {
        "page_width": page_width,
        "page_height": page_height,
        "aspect_ratio": round(aspect_ratio, 4),
        "nodes": len(nodes),
        "visible_nodes": len(visible_nodes),
        "edges": len(edges),
        "text_nodes": len(text_nodes),
        "assets": len(assets),
        "image_tiles": len(image_nodes),
        "ocr_items": int(metadata.get("ocr_items", 0) or 0),
        "max_asset_area_fraction": round(max_asset_area_fraction, 4),
        "total_asset_area_fraction": round(total_asset_area_fraction, 4),
        "full_source_asset": full_source_asset,
        "source_reference_node": source_reference_node,
        "metadata_reference_layer": metadata_reference_layer,
        "blocked_embedding": blocked_embedding,
        "source_embedding_policy": metadata.get("source_embedding_policy"),
        "quality_claim": metadata.get("quality_claim"),
        "reconstruction_mode": metadata.get("gui_reconstruction_mode", metadata.get("reconstruction_mode")),
        "created_by": metadata.get("created_by"),
        "fidelity": metadata.get("fidelity"),
        "architecture_template": metadata.get("architecture_template"),
    }


DIAGNOSTIC_RECONSTRUCTION_MODES = {
    "trace",
    "trace_dense",
    "vector_trace",
    "vector_trace_dense",
    "fallback",
    "dense",
    "high_recall",
}

SEMANTIC_NODE_TYPES = {
    "annotation_block",
    "brace_merge",
    "branch_trunk",
    "caption_block",
    "classifier_head",
    "concat_operator",
    "cuboid_node",
    "dashed_region",
    "feature_map_banded",
    "feature_map_grid",
    "feature_vector_stack",
    "formula_text_block",
    "grid_matrix",
    "group_container",
    "input_output",
    "junction_bus",
    "layer_sequence",
    "loss_region",
    "math_label_box",
    "math_text",
    "math_vector",
    "merge_bus",
    "multi_port_junction",
    "operator_node",
    "polygon_node",
    "process_box",
    "rounded_process",
    "tensor_stack",
    "text_pill",
    "token_grid",
    "trapezoid_node",
}


def semantic_reconstruction_gate(
    scene: dict,
    *,
    mode: str | None,
    no_image_embedding: bool,
) -> dict[str, object]:
    metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
    nodes = [node for node in scene.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in scene.get("edges", []) if isinstance(edge, dict)]
    assets = [asset for asset in scene.get("assets", []) if isinstance(asset, dict)]
    mode_name = str(mode or metadata.get("gui_reconstruction_mode") or metadata.get("reconstruction_mode") or "").lower()
    reconstruction_mode = str(metadata.get("reconstruction_mode") or "").lower()
    created_by = str(metadata.get("created_by") or "")
    created_by_lower = created_by.lower()
    fidelity = str(metadata.get("fidelity") or "")
    fidelity_lower = fidelity.lower()
    architecture_template = str(metadata.get("architecture_template") or "").strip()
    image_tiles = [node for node in nodes if node.get("type") == "image_tile"]
    visible_nodes = [
        node
        for node in nodes
        if node.get("type") not in {"page_background", "audit_region"}
        and not str(node.get("id", "")).lower().endswith("background")
    ]
    text_nodes = [node for node in visible_nodes if str(node.get("text", node.get("symbol", ""))).strip()]
    module_nodes = [node for node in visible_nodes if str(node.get("type") or "") in SEMANTIC_NODE_TYPES]
    icon_parts = sum(1 for node in nodes if node.get("semantic_role") == "editable_icon_polygon")
    icon_parts += sum(1 for edge in edges if edge.get("semantic_role") == "editable_icon_stroke")
    non_icon_line_segments = [
        edge
        for edge in edges
        if edge.get("type") == "line_segment" and edge.get("semantic_role") != "editable_icon_stroke"
    ]

    detail = {
        "mode": mode_name,
        "created_by": created_by,
        "fidelity": fidelity,
        "architecture_template": architecture_template,
        "visible_nodes": len(visible_nodes),
        "semantic_nodes": len(module_nodes),
        "text_nodes": len(text_nodes),
        "edges": len(edges),
        "non_icon_line_segments": len(non_icon_line_segments),
        "icon_parts": icon_parts,
        "assets": len(assets),
        "image_tiles": len(image_tiles),
    }

    def fail(category: str, reason: str) -> dict[str, object]:
        return {"passed": False, "category": category, "reason": reason, **detail}

    def ok(category: str, reason: str) -> dict[str, object]:
        return {"passed": True, "category": category, "reason": reason, **detail}

    if not no_image_embedding:
        return fail("image_embedding", "检测到图片嵌入或原图贴片，不是可编辑模块复现。")
    if assets or image_tiles:
        return fail("raster_tiles", "检测到局部图片贴片；当前 GUI 工作流要求完全可编辑对象。")
    if (
        mode_name in DIAGNOSTIC_RECONSTRUCTION_MODES
        or reconstruction_mode in DIAGNOSTIC_RECONSTRUCTION_MODES
        or ".vector_trace" in created_by_lower
        or fidelity_lower == "auto_editable_vector_trace_draft"
    ):
        return fail("diagnostic_vector_trace", "线稿追踪只是诊断稿，不能算语义模块复现。")
    if architecture_template:
        return ok("semantic_template", f"已匹配类别策略 `{architecture_template}`。")
    if created_by_lower.endswith(".semantic_template") or fidelity_lower == "semantic_editable_rebuild":
        return ok("semantic_template", "已使用语义模板重建。")
    if created_by_lower.endswith(".clean_flow"):
        max_generic_edges = min(240, max(90, len(module_nodes) * 3))
        max_trace_segments = max(40, len(module_nodes) * 2)
        if (
            3 <= len(module_nodes) <= 90
            and (text_nodes or icon_parts >= 12)
            and len(edges) <= max_generic_edges
            and len(non_icon_line_segments) <= max_trace_segments
        ):
            return ok("generic_module_flow", "已生成 OCR/形状锚定的模块化流程图。")
        return fail(
            "weak_generic_flow",
            "只得到弱通用流程草图，模块数、文本锚点或线段密度不符合可交付标准。",
        )
    if created_by_lower == "fig4visio.image_auto_scene":
        return fail("generic_auto_draft", "只得到自动轮廓/图标矢量草图，未匹配到类别策略或稳定模块流。")
    return fail("no_semantic_strategy", "未匹配到可交付的类别策略；不能只靠自动轮廓/线条结果放行。")


def inspect_vsdx_for_images(vsdx_path: Path) -> dict[str, object]:
    media_entries: list[str] = []
    foreign_data_hits: list[str] = []
    image_rel_hits: list[str] = []
    with zipfile.ZipFile(vsdx_path) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if ("/media/" in lower or "/embeddings/" in lower) and lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".emf", ".wmf")):
                media_entries.append(name)
            if not lower.endswith((".xml", ".rels")):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if "ForeignData" in text:
                foreign_data_hits.append(name)
            if re.search(r'(?i)Target="[^"]*(media|embeddings)/', text):
                image_rel_hits.append(name)
    return {
        "media_entries": media_entries,
        "foreign_data_hits": foreign_data_hits,
        "image_rel_hits": image_rel_hits,
        "has_embedded_images": bool(media_entries or foreign_data_hits or image_rel_hits),
    }


def run_self_check(source_image: Path, replica_png: Path, check_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    self_check = load_script_module("self_check")
    check_dir.mkdir(parents=True, exist_ok=True)
    output_json = check_dir / "self_check.json"
    output_png = check_dir / "self_check_comparison.png"
    report = self_check.compare_images(
        source_image,
        replica_png,
        output_json=output_json,
        output_png=output_png,
        threshold=SELF_CHECK_THRESHOLD,
    )
    return output_json, output_png, report


SELF_CHECK_RULE_LABELS = {
    "score_threshold": "总评分",
    "edge_f1": "边缘匹配",
    "foreground_iou": "前景覆盖",
    "grid_density_similarity": "网格密度分布",
    "regional_ink_min_ratio": "分区墨迹覆盖",
    "ink_balance": "整体墨迹比例",
}


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def self_check_failed_rules(report: dict[str, object]) -> list[dict[str, object]]:
    explicit = report.get("failed_rules")
    if isinstance(explicit, list):
        return [item for item in explicit if isinstance(item, dict)]

    metrics = report.get("metrics", {})
    rules = report.get("rules", {})
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(rules, dict):
        rules = {}

    checks = [
        ("score_threshold", "score", report.get("score"), report.get("threshold", rules.get("threshold"))),
        ("edge_f1", "edge_f1", metrics.get("edge_f1"), rules.get("min_edge_f1")),
        ("foreground_iou", "foreground_iou", metrics.get("foreground_iou"), rules.get("min_foreground_iou")),
        (
            "grid_density_similarity",
            "grid_density_similarity",
            metrics.get("grid_density_similarity"),
            rules.get("min_grid_density_similarity"),
        ),
        ("regional_ink_min_ratio", "regional_ink_min_ratio", metrics.get("regional_ink_min_ratio"), rules.get("min_regional_ink_ratio")),
        ("ink_balance", "ink_balance", metrics.get("ink_balance"), rules.get("min_ink_balance")),
    ]
    failures: list[dict[str, object]] = []
    for rule, metric, value_raw, required_raw in checks:
        value = _float_or_none(value_raw)
        required = _float_or_none(required_raw)
        if value is None or required is None:
            continue
        if value < required:
            failures.append({"rule": rule, "metric": metric, "value": value, "required": required})
    return failures


def format_failed_rule(rule: dict[str, object]) -> str:
    rule_name = str(rule.get("rule") or rule.get("metric") or "unknown")
    label = SELF_CHECK_RULE_LABELS.get(rule_name, rule_name)
    value = _float_or_none(rule.get("value"))
    required = _float_or_none(rule.get("required"))
    if value is None or required is None:
        return label
    return f"{label} {value:.3f} < {required:.3f}"


def format_self_check_failure_summary(self_check_report: dict[str, object], attempts: list[dict[str, object]]) -> str:
    score = float(self_check_report.get("score", 0.0) or 0.0)
    threshold = float(self_check_report.get("threshold", SELF_CHECK_THRESHOLD) or SELF_CHECK_THRESHOLD)
    failures = self_check_failed_rules(self_check_report)
    failure_text = "；".join(format_failed_rule(rule) for rule in failures[:4])
    if len(failures) > 4:
        failure_text += f"；另有 {len(failures) - 4} 项未达标"

    if score < threshold:
        if failure_text:
            return f"已自动重跑 {len(attempts)} 轮；最佳截图评分 {score:.3f} 低于阈值 {threshold:.2f}，未达标项：{failure_text}。"
        return f"已自动重跑 {len(attempts)} 轮；最佳截图评分 {score:.3f} 低于阈值 {threshold:.2f}。"
    if failure_text:
        return f"已自动重跑 {len(attempts)} 轮；最佳截图评分 {score:.3f} 已达到阈值 {threshold:.2f}，但结构自检门槛未通过：{failure_text}。"
    return f"已自动重跑 {len(attempts)} 轮；最佳截图评分 {score:.3f} 已达到阈值 {threshold:.2f}，但自检状态仍为失败，请查看对比图。"


def format_semantic_gate_failure_summary(
    semantic_gate: dict[str, object],
    self_check_report: dict[str, object],
    attempts: list[dict[str, object]],
) -> str:
    score = float(self_check_report.get("score", 0.0) or 0.0)
    threshold = float(self_check_report.get("threshold", SELF_CHECK_THRESHOLD) or SELF_CHECK_THRESHOLD)
    reason = str(semantic_gate.get("reason") or "未达到语义模块复现门槛")
    category = str(semantic_gate.get("category") or "semantic_gate_failed")
    if score >= threshold:
        return f"已自动重跑 {len(attempts)} 轮；最佳截图评分 {score:.3f} 已达到阈值 {threshold:.2f}，但模块复现门槛未通过（{category}）：{reason}"
    return f"已自动重跑 {len(attempts)} 轮；最佳截图评分 {score:.3f} 低于阈值 {threshold:.2f}，且模块复现门槛未通过（{category}）：{reason}"


def format_forced_output_summary(
    semantic_gate: dict[str, object],
    self_check_report: dict[str, object],
    attempts: list[dict[str, object]],
) -> str:
    score = float(self_check_report.get("score", 0.0) or 0.0)
    threshold = float(self_check_report.get("threshold", SELF_CHECK_THRESHOLD) or SELF_CHECK_THRESHOLD)
    category = str(semantic_gate.get("category") or "semantic_gate_unknown")
    reason = str(semantic_gate.get("reason") or "模块门槛未完全通过")
    if score >= threshold and bool(semantic_gate.get("passed")):
        return f"已自动重跑 {len(attempts)} 轮；当前结果已达到下载条件，评分 {score:.3f}。"
    return (
        f"已自动重跑 {len(attempts)} 轮；最佳结果仍未完全达标，截图评分 {score:.3f} / 阈值 {threshold:.2f}；"
        f"模块门槛={category}。按当前工作流强制输出现有文件供下载，请人工复核后使用。原因：{reason}"
    )


def write_quality_report(
    *,
    scene_path: Path,
    source_image: Path,
    output_dir: Path,
    review_dir: Path,
    review_manifest: Path,
    quality_dir: Path,
    complexity_report: Path,
    audit_report: Path,
    self_check_json: Path,
    self_check_png: Path,
    self_check_report: dict[str, object],
    attempts: list[dict[str, object]],
    download_allowed: bool,
    no_image_embedding: bool,
    semantic_gate: dict[str, object],
    forced_output: bool = False,
) -> tuple[Path, Path, str, str, str]:
    quality_dir.mkdir(parents=True, exist_ok=True)
    scene = read_json(scene_path)
    metrics = scene_quality_metrics(scene, source_image)
    score = float(self_check_report.get("score", 0.0) or 0.0)

    if not no_image_embedding:
        status = "blocked_embedding_detected"
        label = "未通过：检测到图片嵌入"
        summary = "本次结果含有图片嵌入痕迹，不允许下载。"
    elif forced_output:
        status = "forced_output_after_retries"
        label = "强制输出：未达标但可下载"
        summary = format_forced_output_summary(semantic_gate, self_check_report, attempts)
    elif not bool(semantic_gate.get("passed")):
        status = "semantic_gate_failed"
        label = "未通过：不是模块复现"
        summary = format_semantic_gate_failure_summary(semantic_gate, self_check_report, attempts)
    elif download_allowed:
        status = "self_check_passed"
        label = "自检通过：可以下载"
        summary = f"截图自检通过，评分 {score:.3f}，已确认没有图片嵌入，并通过模块复现门槛。"
    else:
        status = "self_check_failed"
        label = "自检未通过：禁止下载"
        summary = format_self_check_failure_summary(self_check_report, attempts)

    report = {
        "schema_version": "0.2",
        "status": status,
        "label": label,
        "summary": summary,
        "download_allowed": download_allowed,
        "forced_output": forced_output,
        "metrics": metrics,
        "self_check": self_check_report,
        "semantic_gate": semantic_gate,
        "attempts": attempts,
        "paths": {
            "source_image": str(source_image.resolve()),
            "scene": str(scene_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "review_dir": str(review_dir.resolve()),
            "review_manifest": str(review_manifest.resolve()),
            "complexity_report": str(complexity_report.resolve()),
            "audit_report": str(audit_report.resolve()),
            "self_check_json": str(self_check_json.resolve()),
            "self_check_png": str(self_check_png.resolve()),
        },
        "workflow": [
            "stage_source_image",
            "generate_editable_scene_without_image_embedding",
            "render_visio_outputs",
            "screenshot_self_check",
            "semantic_module_gate",
            "retry_with_vector_trace_as_diagnostic_only",
            "allow_download_only_after_screenshot_and_semantic_gate_pass",
            "force_output_after_five_attempts_when_no_embedding",
        ],
    }
    quality_json = quality_dir / "quality_report.json"
    write_json(quality_json, report)

    md_lines = [
        "# Fig4Visio Quality Report",
        "",
        f"- 状态: {label}",
        f"- 结论: {summary}",
        f"- 是否允许下载: {download_allowed}",
        f"- 是否强制输出: {forced_output}",
        f"- 模块复现门槛: {semantic_gate.get('category')} / {semantic_gate.get('reason')}",
        f"- 自检对比图: `{self_check_png}`",
        f"- Scene: `{scene_path}`",
        "",
        "## Self Check",
    ]
    for key, value in self_check_report.get("metrics", {}).items():
        md_lines.append(f"- {key}: {value}")
    failed_rules = self_check_failed_rules(self_check_report)
    if failed_rules:
        md_lines.extend(["", "## Failed Rules"])
        for rule in failed_rules:
            md_lines.append(f"- {format_failed_rule(rule)}")
    md_lines.extend(["", "## Attempts"])
    for attempt in attempts:
        md_lines.append(
            f"- round {attempt.get('round')}: mode={attempt.get('mode')}, "
            f"score={attempt.get('self_check_score')}, pass={attempt.get('passed')}, "
            f"no_embed={attempt.get('no_image_embedding')}, "
            f"semantic={attempt.get('semantic_gate_passed')}:{attempt.get('semantic_gate_category')}"
        )
    quality_md = quality_dir / "quality_report.md"
    quality_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return quality_json, quality_md, status, label, summary


def run_attempt(
    *,
    attempt_index: int,
    mode: str,
    run_dir: Path,
    staged_source: Path,
    basename: str,
    log,
) -> dict[str, object]:
    attempt_dir = run_dir / f"attempt_{attempt_index:02d}_{mode}"
    output_dir = attempt_dir / "exports"
    check_dir = attempt_dir / "self_check"
    scene_path = attempt_dir / f"{basename}.scene.json"

    log(f"第 {attempt_index} 轮生成：{mode}")
    auto_scene_for_image(staged_source, scene_path, reconstruction_mode=mode)

    log("校验 scene 结构...")
    log(validate_scene_file(scene_path, strict=False))

    log("调用 Microsoft Visio 渲染截图和 VSDX...")
    log(render_scene(scene_path, output_dir, basename, visible=False))

    vsdx_path = output_dir / f"{basename}.vsdx"
    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    missing = [path.name for path in (vsdx_path, png_path, svg_path) if not path.exists()]
    if missing:
        raise RuntimeError("Visio 渲染完成但缺少输出文件: " + ", ".join(missing))

    scene = read_json(scene_path)
    scene_info = scene_quality_metrics(scene, staged_source)
    vsdx_info = inspect_vsdx_for_images(vsdx_path)
    no_image_embedding = not scene_info["blocked_embedding"] and not bool(vsdx_info["has_embedded_images"])
    semantic_gate = semantic_reconstruction_gate(scene, mode=mode, no_image_embedding=no_image_embedding)

    log("运行截图自检...")
    self_check_json, self_check_png, self_report = run_self_check(staged_source, png_path, check_dir)
    passed = bool(no_image_embedding and self_report.get("passed") and semantic_gate.get("passed"))
    failed_rule_text = "；".join(format_failed_rule(rule) for rule in self_check_failed_rules(self_report)[:4])
    log(
        f"自检结果: {'通过' if passed else '失败'}; "
        f"score={float(self_report.get('score', 0.0)):.3f}; "
        f"no_image_embedding={no_image_embedding}; "
        f"semantic_gate={semantic_gate.get('category')}"
        + (f"; 未达标项={failed_rule_text}" if failed_rule_text else "")
        + (f"; 模块门槛={semantic_gate.get('reason')}" if not bool(semantic_gate.get("passed")) else "")
    )

    return {
        "round": attempt_index,
        "mode": mode,
        "attempt_dir": attempt_dir,
        "scene": scene_path,
        "output_dir": output_dir,
        "vsdx": vsdx_path,
        "png": png_path,
        "svg": svg_path,
        "self_check_json": self_check_json,
        "self_check_png": self_check_png,
        "self_check_report": self_report,
        "self_check_score": float(self_report.get("score", 0.0) or 0.0),
        "self_check_passed": bool(self_report.get("passed")),
        "no_image_embedding": no_image_embedding,
        "semantic_gate": semantic_gate,
        "semantic_gate_passed": bool(semantic_gate.get("passed")),
        "semantic_gate_category": semantic_gate.get("category"),
        "semantic_gate_reason": semantic_gate.get("reason"),
        "scene_info": scene_info,
        "vsdx_info": vsdx_info,
        "passed": passed,
    }


def attempt_score(attempt: dict[str, object]) -> float:
    return float(attempt.get("self_check_score", 0.0) or 0.0)


def reconstruction_modes() -> list[str]:
    return RECONSTRUCTION_MODE_SEQUENCE[:MAX_AUTO_ATTEMPTS]


def select_attempt_for_delivery(attempts: list[dict[str, object]]) -> tuple[dict[str, object] | None, str]:
    for attempt in attempts:
        if attempt.get("passed"):
            return attempt, "passed"
    semantic_attempts = [attempt for attempt in attempts if attempt.get("semantic_gate_passed")]
    if semantic_attempts:
        return max(semantic_attempts, key=attempt_score), "best_semantic_failed_self_check"
    if attempts:
        return max(attempts, key=attempt_score), "best_diagnostic_no_semantic"
    return None, "none"


def should_force_output_after_retries(attempts: list[dict[str, object]], selected: dict[str, object] | None) -> bool:
    if selected is None or selected.get("passed"):
        return False
    return len(attempts) >= MAX_AUTO_ATTEMPTS and bool(selected.get("no_image_embedding"))


def run_fig4visio_job(source_path: Path, *, log) -> OutputSet:
    run_root = Path.cwd() / "work" / "gui_runs"
    run_dir = run_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    basename = safe_stem(source_path)

    log("固定本次原图，建立可复核的 source/original 文件...")
    staged_source, source_manifest = stage_source_image(source_path, run_dir, basename)
    log(f"原图已固定: {staged_source}")

    modes = reconstruction_modes()
    attempts: list[dict[str, object]] = []

    for index, mode in enumerate(modes, 1):
        attempt = run_attempt(
            attempt_index=index,
            mode=mode,
            run_dir=run_dir,
            staged_source=staged_source,
            basename=basename,
            log=log,
        )
        attempts.append(attempt)
        if attempt["passed"]:
            log(f"第 {index} 轮已通过自检，停止重跑。")
            break
        if index < len(modes):
            if index == INITIAL_AUTO_ATTEMPTS:
                log("前三轮仍未达到下载条件，继续追加两轮复现尝试...")
            else:
                log("截图差距过大，自动切换策略重跑一轮...")

    selected, selection_reason = select_attempt_for_delivery(attempts)
    if selected is None:
        raise RuntimeError("没有生成任何可评估的输出。")
    if not selected.get("passed"):
        if selection_reason == "best_semantic_failed_self_check":
            log("所有自动轮次均未同时通过截图和模块门槛；保留最佳模块复现轮次。")
        elif selection_reason == "best_diagnostic_no_semantic":
            log("没有生成合格模块复现轮次；仅保留诊断轮次。")
        else:
            log("所有自动轮次均未通过；保留最佳轮次。")

    attempt_dir = Path(selected["attempt_dir"])
    scene_path = Path(selected["scene"])
    output_dir = Path(selected["output_dir"])
    review_dir = attempt_dir / "review"
    quality_dir = attempt_dir / "quality"
    png_path = Path(selected["png"])

    log("生成复杂度报告和模块审计报告...")
    complexity_report, audit_report, report_log = write_scene_reports(scene_path, quality_dir)
    log(report_log)

    log("生成原图/输出图复核包...")
    review_manifest, pair_preview, review_log = make_review_bundle(staged_source, png_path, scene_path, review_dir, basename)
    log(review_log)

    no_image_embedding = bool(selected["no_image_embedding"])
    forced_output = should_force_output_after_retries(attempts, selected)
    download_allowed = bool(selected["passed"] or forced_output)
    if forced_output:
        log("5 轮后仍未完全达标；按设置强制输出现有最佳文件，并在质量报告中标记未达标。")
    quality_json, quality_md, status, label, summary = write_quality_report(
        scene_path=scene_path,
        source_image=staged_source,
        output_dir=output_dir,
        review_dir=review_dir,
        review_manifest=review_manifest,
        quality_dir=quality_dir,
        complexity_report=complexity_report,
        audit_report=audit_report,
        self_check_json=Path(selected["self_check_json"]),
        self_check_png=Path(selected["self_check_png"]),
        self_check_report=selected["self_check_report"],
        attempts=attempts,
        download_allowed=download_allowed,
        no_image_embedding=no_image_embedding,
        semantic_gate=selected["semantic_gate"],
        forced_output=forced_output,
    )
    log(f"{label}: {summary}")

    return OutputSet(
        run_dir=run_dir,
        attempt_dir=attempt_dir,
        source_image=staged_source,
        source_manifest=source_manifest,
        scene=scene_path,
        output_dir=output_dir,
        basename=basename,
        vsdx=Path(selected["vsdx"]),
        png=png_path,
        svg=Path(selected["svg"]),
        review_dir=review_dir,
        review_manifest=review_manifest,
        quality_json=quality_json,
        quality_md=quality_md,
        complexity_report=complexity_report,
        audit_report=audit_report,
        pair_preview=pair_preview,
        self_check_json=Path(selected["self_check_json"]),
        self_check_png=Path(selected["self_check_png"]),
        self_check_passed=bool(selected["self_check_passed"]),
        self_check_score=float(selected["self_check_score"]),
        download_allowed=download_allowed,
        quality_status=status,
        quality_label=label,
        quality_summary=summary,
    )


class Fig4VisioGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.source_path: Path | None = None
        self.outputs: OutputSet | None = None
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.input_preview_image: ImageTk.PhotoImage | None = None
        self.output_preview_image: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="上传图片后自动拆分、截图自检，通过后才允许下载")
        self.file_var = tk.StringVar(value="未选择文件")

        self._build_ui()
        self.after(120, self._drain_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, width=310)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Fig4Visio", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self.file_var, wraplength=290).grid(row=1, column=0, sticky="ew", pady=(8, 14))

        self.process_button = ttk.Button(sidebar, text="上传图片并拆分", command=self.pick_image)
        self.process_button.grid(row=2, column=0, sticky="ew")

        downloads = ttk.LabelFrame(sidebar, text="下载", padding=10)
        downloads.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        downloads.columnconfigure(0, weight=1)
        self.save_vsdx_button = ttk.Button(downloads, text="下载 Visio 文件", command=lambda: self.save_output("vsdx"), state=tk.DISABLED)
        self.save_png_button = ttk.Button(downloads, text="下载预览图", command=lambda: self.save_output("png"), state=tk.DISABLED)
        for index, button in enumerate([self.save_vsdx_button, self.save_png_button]):
            button.grid(row=index, column=0, sticky="ew", pady=(0 if index == 0 else 6, 0))

        ttk.Button(sidebar, text="打开输出目录", command=self.open_output_dir).grid(row=4, column=0, sticky="ew", pady=(14, 0))
        ttk.Label(
            sidebar,
            text="工作流：上传图片 -> 可编辑拆分 -> 渲染截图 -> 自动对比原图 -> 失败自动重跑 -> 通过后开放下载。默认不嵌入图片。",
            wraplength=290,
            foreground="#475569",
        ).grid(row=5, column=0, sticky="ew", pady=(16, 0))
        ttk.Label(sidebar, textvariable=self.status_var, wraplength=290).grid(row=6, column=0, sticky="ew", pady=(10, 0))
        sidebar.rowconfigure(7, weight=1)

        self.progress = ttk.Progressbar(sidebar, mode="indeterminate")
        self.progress.grid(row=8, column=0, sticky="ew", pady=(12, 0))

        main = ttk.Notebook(root)
        main.grid(row=0, column=1, sticky="nsew")

        self.input_frame = ttk.Frame(main, padding=10)
        self.output_frame = ttk.Frame(main, padding=10)
        self.log_frame = ttk.Frame(main, padding=10)
        main.add(self.input_frame, text="输入预览")
        main.add(self.output_frame, text="输出预览")
        main.add(self.log_frame, text="处理日志")

        self.input_frame.rowconfigure(0, weight=1)
        self.input_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)
        self.output_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(0, weight=1)
        self.log_frame.columnconfigure(0, weight=1)

        self.input_preview = ttk.Label(self.input_frame, anchor="center", text="点击左侧按钮上传图片")
        self.input_preview.grid(row=0, column=0, sticky="nsew")

        self.output_preview = ttk.Label(self.output_frame, anchor="center", text="处理完成后显示 PNG 预览")
        self.output_preview.grid(row=0, column=0, sticky="nsew")
        self.output_text = ScrolledText(self.output_frame, height=10, wrap=tk.WORD)
        self.output_text.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.output_text.configure(state=tk.DISABLED)

        self.log_text = ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state=tk.DISABLED)

    def pick_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.set_source(Path(path))
            self.start_processing()

    def set_source(self, path: Path) -> None:
        self.source_path = path
        self.outputs = None
        self.file_var.set(str(path))
        self.status_var.set("已选择图片，准备自动拆分")
        self._set_output_buttons(False)
        self.output_preview.configure(image="", text="处理完成后显示 PNG 预览")
        self.output_preview_image = None
        self._set_output_text("")
        self._append_log(f"选择文件: {path}\n")
        self._show_image(path, target="input")

    def _show_image(self, path: Path, *, target: str) -> None:
        try:
            with Image.open(path) as image:
                preview = image.copy()
                preview.thumbnail(MAX_PREVIEW_SIZE)
            photo = ImageTk.PhotoImage(preview)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"无法预览图片:\n{exc}")
            return

        if target == "input":
            self.input_preview_image = photo
            self.input_preview.configure(image=photo, text="")
            self.input_preview.grid(row=0, column=0, sticky="nsew")
        else:
            self.output_preview_image = photo
            self.output_preview.configure(image=photo, text="")

    def start_processing(self) -> None:
        if self.source_path is None:
            messagebox.showwarning(APP_NAME, "请先上传图片。")
            return
        self.process_button.configure(state=tk.DISABLED)
        self.progress.start(10)
        self.status_var.set("正在处理并自检...")
        self._set_output_buttons(False)
        self._append_log("\n--- 开始处理 ---\n")

        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

    def _worker(self) -> None:
        assert self.source_path is not None

        def log(message: str) -> None:
            self.queue.put(("log", str(message).rstrip() + "\n"))

        try:
            pythoncom = None
            try:
                pythoncom = importlib.import_module("pythoncom")
                pythoncom.CoInitialize()
            except Exception:
                pythoncom = None
            try:
                outputs = run_fig4visio_job(self.source_path, log=log)
            finally:
                if pythoncom is not None:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
            self.queue.put(("done", outputs))
        except Exception as exc:
            error = f"{exc}\n\n{traceback.format_exc()}"
            self.queue.put(("error", error))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(str(payload))
        except queue.Empty:
            pass
        self.after(120, self._drain_queue)

    def _on_done(self, outputs: OutputSet) -> None:
        self.outputs = outputs
        self.progress.stop()
        self.process_button.configure(state=tk.NORMAL)
        self.status_var.set(outputs.quality_label)
        self._set_output_buttons(outputs.download_allowed)
        self._show_image(outputs.png, target="output")
        self._set_output_text(
            "\n".join(
                [
                    f"质量状态: {outputs.quality_label}",
                    f"说明: {outputs.quality_summary}",
                    f"自检评分: {outputs.self_check_score:.3f}",
                    f"允许下载: {'是' if outputs.download_allowed else '否'}",
                    "",
                    f"VSDX: {outputs.vsdx}",
                    f"PNG:  {outputs.png}",
                    f"自检对比图: {outputs.self_check_png}",
                    f"质量报告: {outputs.quality_md}",
                    f"输出目录: {outputs.output_dir}",
                ]
            )
        )
        self._append_log("--- 处理完成 ---\n")

    def _on_error(self, error: str) -> None:
        self.progress.stop()
        self.process_button.configure(state=tk.NORMAL)
        self.status_var.set("处理失败")
        self._append_log(error + "\n")
        messagebox.showerror(APP_NAME, "处理失败，详细信息请查看处理日志。")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_output_text(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state=tk.DISABLED)

    def _set_output_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (self.save_vsdx_button, self.save_png_button):
            button.configure(state=state)

    def save_output(self, kind: str) -> None:
        if self.outputs is None or not self.outputs.download_allowed:
            messagebox.showwarning(APP_NAME, "自检未通过，当前结果不允许下载。")
            return
        mapping = {
            "vsdx": (self.outputs.vsdx, "Visio 文件", ".vsdx"),
            "png": (self.outputs.png, "PNG 图片", ".png"),
        }
        source, label, suffix = mapping[kind]
        self._save_copy(source, label, suffix)

    def _save_copy(self, source: Path, label: str, suffix: str) -> None:
        if not source.exists():
            messagebox.showerror(APP_NAME, f"文件不存在:\n{source}")
            return
        target = filedialog.asksaveasfilename(
            title=f"保存{label}",
            initialfile=source.name,
            defaultextension=suffix,
            filetypes=[(label, f"*{suffix}"), ("All files", "*.*")],
        )
        if not target:
            return
        shutil.copy2(source, target)
        self.status_var.set(f"已保存: {target}")

    def open_output_dir(self) -> None:
        path = self.outputs.attempt_dir if self.outputs else Path.cwd() / "work" / "gui_runs"
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"无法打开目录:\n{exc}")


def main() -> None:
    if "--smoke" in sys.argv:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="fig4visio_gui_smoke_") as temp_dir:
            temp_root = Path(temp_dir)
            image_path = temp_root / "smoke.png"
            image = Image.new("RGB", (320, 180), "white")
            image.save(image_path)
            run_fig4visio_job(image_path, log=lambda _: None)
        raise SystemExit(0)
    app = Fig4VisioGui()
    app.mainloop()


if __name__ == "__main__":
    main()
