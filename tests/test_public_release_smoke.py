from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
from PIL import ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import image_auto_scene  # noqa: E402
import self_check  # noqa: E402
import gui_app  # noqa: E402
from scene_to_visio import edge_route_points, edge_style, load_component_map, normalize_scene_coordinates, rounded_orthogonal_points  # noqa: E402


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def draw_synthetic_icon_flow(path: Path) -> None:
    image = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(image)
    stroke = "#1F2937"
    blue = "#2563EB"
    green = "#059669"
    orange = "#EA580C"

    boxes = [(42, 120, 214, 250), (356, 120, 544, 250), (682, 120, 858, 250)]
    for box in boxes:
        draw.rounded_rectangle(box, radius=16, outline="#94A3B8", width=3, fill="#F8FAFC")
    draw.line((214, 185, 356, 185), fill=stroke, width=4)
    draw.polygon([(356, 185), (340, 176), (340, 194)], fill=stroke)
    draw.line((544, 185, 682, 185), fill=stroke, width=4)
    draw.polygon([(682, 185), (666, 176), (666, 194)], fill=stroke)

    # Cloud icon.
    draw.arc((72, 158, 132, 222), 188, 358, fill=blue, width=5)
    draw.arc((116, 132, 188, 220), 190, 345, fill=blue, width=5)
    draw.arc((150, 164, 212, 222), 200, 358, fill=blue, width=5)
    draw.line((86, 220, 196, 220), fill=blue, width=5)

    # Database cylinder icon.
    draw.ellipse((394, 136, 504, 182), outline=green, width=5)
    draw.line((394, 158, 394, 226), fill=green, width=5)
    draw.line((504, 158, 504, 226), fill=green, width=5)
    draw.arc((394, 204, 504, 248), 0, 180, fill=green, width=5)
    draw.arc((394, 170, 504, 214), 0, 180, fill=green, width=4)

    # User plus magnifier icon.
    draw.ellipse((720, 136, 766, 182), outline=orange, width=5)
    draw.arc((704, 174, 784, 246), 200, 340, fill=orange, width=5)
    draw.ellipse((798, 146, 842, 190), outline=stroke, width=5)
    draw.line((833, 181, 860, 208), fill=stroke, width=5)

    image.save(path)


def fake_ocr_items(texts: list[str]) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "text": text,
            "confidence": 0.98,
            "box": image_auto_scene.Box(10 + index * 8, 10, 40, 20),
            "points": [[10, 10], [50, 10], [50, 30], [10, 30]],
        }
        for index, text in enumerate(texts)
    ]


def draw_swin_like_line_art(path: Path, *, omit_left_pipeline: bool = False) -> None:
    image = Image.new("RGB", (1148, 355), "white")
    draw = ImageDraw.Draw(image)
    stroke = "#111111"
    if not omit_left_pipeline:
        draw.rectangle((6, 184, 72, 229), outline=stroke, width=2)
        draw.rectangle((91, 132, 120, 258), outline=stroke, width=2)
        stages = [(132, 102, 303, 300), (312, 102, 474, 300), (483, 102, 643, 300), (654, 102, 814, 300)]
        for sx, sy, ex, ey in stages:
            draw.rounded_rectangle((sx, sy, ex, ey), radius=22, outline=stroke, width=2)
            draw.rectangle((sx + 20, 132, sx + 49, 258), outline=stroke, width=2)
            draw.rounded_rectangle((sx + 70, 132, sx + 156, 278), radius=14, outline=stroke, width=2)
        for x1, x2 in [(72, 91), (120, 151), (180, 202), (288, 326), (355, 374), (460, 497), (526, 544), (630, 668), (697, 716), (802, 824)]:
            draw.line((x1, 206, x2, 206), fill=stroke, width=2)
            draw.polygon([(x2, 206), (x2 - 9, 201), (x2 - 9, 211)], fill=stroke)
    for x in (862, 1018):
        draw.rounded_rectangle((x, 19, x + 111, 301), radius=18, outline=stroke, width=2)
        for y in (64, 109, 194, 241):
            draw.rectangle((x + 13, y, x + 77, y + 28), outline=stroke, width=2)
        for y in (40, 169):
            draw.ellipse((x + 31, y - 11, x + 54, y + 12), outline=stroke, width=2)
        draw.line((x + 45, 286, x + 45, 52), fill=stroke, width=2)
    image.save(path)


def test_public_release_files_are_present() -> None:
    for rel in [
        "SKILL.md",
        "README.md",
        "LICENSE",
        "requirements.txt",
        "sync_to_skill.py",
        "references/review-contract.md",
        "references/reviewer-two-image-prompt.md",
        "references/full-scene-regeneration-prompt.md",
        "references/renderer-effective-fields.json",
        "scripts/make_review_assets.py",
        "scripts/review_checklist_gate.py",
        "scripts/review_findings_to_repair_plan.py",
        "scripts/prepare_regeneration_packet.py",
        "scripts/round_noop_gate.py",
    ]:
        assert (ROOT / rel).exists(), rel


def test_basic_scene_validates() -> None:
    result = run_script("scene_validate.py", str(ROOT / "templates" / "examples" / "basic_flow.scene.json"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Scene is valid" in result.stdout


def test_image_auto_scene_reconstructs_icons_as_editable_vectors(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "icon_flow.png"
    draw_synthetic_icon_flow(source)
    monkeypatch.setattr(image_auto_scene, "run_ocr", lambda _path: [])

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    metadata = scene["metadata"]
    icon_edges = [edge for edge in scene["edges"] if edge.get("semantic_role") == "editable_icon_stroke"]
    icon_nodes = [node for node in scene["nodes"] if node.get("semantic_role") == "editable_icon_polygon"]

    assert metadata["icon_reconstruction_policy"] == "editable_vector_no_raster"
    assert metadata["icon_vector_regions"] >= 3
    assert len(icon_edges) + len(icon_nodes) >= 30
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert all(edge["type"] == "line_segment" for edge in icon_edges)


def test_vector_trace_mode_keeps_icon_vectors_no_raster(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "icon_flow_trace.png"
    draw_synthetic_icon_flow(source)
    monkeypatch.setattr(image_auto_scene, "run_ocr", lambda _path: [])

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="vector_trace_dense")
    icon_edges = [edge for edge in scene["edges"] if edge.get("semantic_role") == "editable_icon_stroke"]

    assert scene["metadata"]["icon_vector_regions"] >= 3
    assert len(icon_edges) >= 30
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])


def test_gui_semantic_gate_blocks_vector_trace_even_with_good_score() -> None:
    scene = {
        "version": "0.1",
        "metadata": {
            "created_by": "fig4visio.image_auto_scene.vector_trace_dense",
            "fidelity": "auto_editable_vector_trace_draft",
            "reconstruction_mode": "vector_trace_dense",
        },
        "page": {"width": 900, "height": 420, "units": "px"},
        "nodes": [
            {"id": "page_background", "type": "page_background", "x": 0, "y": 0, "w": 900, "h": 420},
            {"id": "label", "type": "text_block", "x": 40, "y": 40, "w": 120, "h": 30, "text": "Input"},
        ],
        "edges": [{"id": "trace_001", "type": "line_segment", "points": [[10, 10], [200, 10]]}],
        "assets": [],
    }

    gate = gui_app.semantic_reconstruction_gate(scene, mode="vector_trace_dense", no_image_embedding=True)

    assert gate["passed"] is False
    assert gate["category"] == "diagnostic_vector_trace"
    assert "线稿追踪" in gate["reason"]


def test_gui_semantic_gate_accepts_category_template() -> None:
    scene = {
        "version": "0.1",
        "metadata": {
            "created_by": "fig4visio.image_auto_scene.remote_sensing_rsei_workflow",
            "fidelity": "semantic_editable_rebuild",
            "architecture_template": "remote_sensing_rsei_workflow",
            "reconstruction_mode": "standard",
        },
        "page": {"width": 900, "height": 420, "units": "px"},
        "nodes": [
            {"id": "page_background", "type": "page_background", "x": 0, "y": 0, "w": 900, "h": 420},
            {"id": "input", "type": "process_box", "x": 40, "y": 40, "w": 120, "h": 60, "text": "Input"},
            {"id": "process", "type": "process_box", "x": 260, "y": 40, "w": 120, "h": 60, "text": "Process"},
        ],
        "edges": [{"id": "flow", "type": "arrow_connector", "from": "input:right", "to": "process:left"}],
        "assets": [],
    }

    gate = gui_app.semantic_reconstruction_gate(scene, mode="standard", no_image_embedding=True)

    assert gate["passed"] is True
    assert gate["category"] == "semantic_template"


def test_gui_semantic_gate_rejects_dense_generic_clean_flow() -> None:
    scene = {
        "version": "0.1",
        "metadata": {
            "created_by": "fig4visio.image_auto_scene.clean_flow",
            "fidelity": "generic_clean_flow_editable_rebuild",
            "reconstruction_mode": "standard",
        },
        "page": {"width": 1200, "height": 800, "units": "px"},
        "nodes": [
            {"id": "page_background", "type": "page_background", "x": 0, "y": 0, "w": 1200, "h": 800},
        ]
        + [
            {"id": f"box_{index}", "type": "process_box", "x": 20 + index * 10, "y": 20, "w": 60, "h": 30, "text": f"N{index}"}
            for index in range(100)
        ],
        "edges": [
            {
                "id": f"edge_{index}",
                "type": "arrow_connector",
                "from": f"box_{index % 100}:right",
                "to": f"box_{(index + 1) % 100}:left",
            }
            for index in range(250)
        ],
        "assets": [],
    }

    gate = gui_app.semantic_reconstruction_gate(scene, mode="standard", no_image_embedding=True)

    assert gate["passed"] is False
    assert gate["category"] == "weak_generic_flow"


def test_gui_selects_semantic_attempt_before_diagnostic_trace() -> None:
    attempts = [
        {
            "round": 1,
            "self_check_score": 0.31,
            "passed": False,
            "semantic_gate_passed": True,
            "mode": "standard",
        },
        {
            "round": 2,
            "self_check_score": 0.66,
            "passed": False,
            "semantic_gate_passed": False,
            "mode": "vector_trace",
        },
    ]

    selected, reason = gui_app.select_attempt_for_delivery(attempts)

    assert selected is attempts[0]
    assert reason == "best_semantic_failed_self_check"


def test_gui_retry_sequence_runs_five_rounds_after_initial_three() -> None:
    modes = gui_app.reconstruction_modes()

    assert len(modes) == 5
    assert modes[:3] == ["standard", "vector_trace", "vector_trace_dense"]
    assert modes[3:] == ["standard", "vector_trace_dense"]


def test_gui_forces_download_after_five_failed_non_embedded_attempts() -> None:
    attempts = [
        {
            "round": index + 1,
            "passed": False,
            "self_check_score": 0.2 + index * 0.01,
            "no_image_embedding": True,
            "semantic_gate_passed": False,
        }
        for index in range(gui_app.MAX_AUTO_ATTEMPTS)
    ]
    selected = attempts[-1]

    assert gui_app.should_force_output_after_retries(attempts, selected) is True


def test_gui_does_not_force_download_when_embedding_detected() -> None:
    attempts = [
        {"round": index + 1, "passed": False, "self_check_score": 0.2, "no_image_embedding": False}
        for index in range(gui_app.MAX_AUTO_ATTEMPTS)
    ]

    assert gui_app.should_force_output_after_retries(attempts, attempts[-1]) is False


def test_gui_forced_output_summary_keeps_failure_visible() -> None:
    summary = gui_app.format_forced_output_summary(
        {"passed": False, "category": "diagnostic_vector_trace", "reason": "线稿追踪只是诊断稿"},
        {"score": 0.31, "threshold": 0.38},
        [{} for _ in range(gui_app.MAX_AUTO_ATTEMPTS)],
    )

    assert "强制输出" in summary
    assert "未完全达标" in summary
    assert "diagnostic_vector_trace" in summary


def test_swin_transformer_architecture_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "swin_arch.png"
    Image.new("RGB", (1148, 355), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Swin Transformer Block",
            "Stage 1",
            "Patch Partition",
            "Patch Merging",
            "W-MSA",
            "SW-MSA",
            "MLP",
            "LN",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.swin_transformer_architecture"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert sum(1 for node in scene["nodes"] if str(node.get("id", "")).endswith("_frame")) >= 6
    assert "Patch Partition" in texts
    assert "Linear Embedding" in texts
    assert "Swin\nTransformer\nBlock" in texts
    assert "W-MSA" in texts and "SW-MSA" in texts
    assert len(scene["edges"]) >= 30


def test_sparse_swin_transformer_variant_uses_no_frame_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sparse_swin_arch.png"
    Image.new("RGB", (1996, 622), "white").save(source)

    def sparse_ocr(_path: Path) -> list[dict[str, object]]:
        texts = [
            (24, 354, 61, 25, "Images"),
            (329, 188, 45, 21, "Stage"),
            (396, 323, 50, 27, "Swin"),
            (367, 351, 109, 23, "Transformer"),
            (393, 374, 56, 24, "Block"),
            (576, 339, 57, 21, "PatchMergi"),
            (639, 188, 50, 21, "Stage2"),
            (698, 323, 53, 26, "Swin"),
            (669, 350, 112, 23, "Transformer"),
            (695, 374, 59, 24, "Block"),
            (879, 339, 57, 21, "PatchMergi"),
            (943, 189, 37, 19, "Stage"),
            (1000, 323, 51, 26, "Swin"),
            (971, 352, 110, 21, "Transformer"),
            (997, 375, 57, 24, "Block"),
            (1181, 340, 57, 21, "PatchMergi"),
            (1305, 323, 52, 27, "Swin"),
            (1276, 351, 111, 23, "Transformer"),
            (1302, 374, 57, 24, "Block"),
            (1553, 112, 68, 29, "MLP"),
            (1564, 194, 47, 28, "LN"),
            (1556, 355, 61, 20, "W-MSA"),
            (1563, 427, 48, 29, "LN"),
            (1831, 111, 68, 29, "MLP"),
            (1841, 194, 48, 28, "LN"),
            (1832, 356, 65, 20, "SW-MSA"),
            (1839, 425, 51, 34, "LN"),
        ]
        items: list[dict[str, object]] = []
        for index, (x, y, w, h, text) in enumerate(texts):
            items.append(
                {
                    "id": index,
                    "text": text,
                    "confidence": 0.98,
                    "box": image_auto_scene.Box(x, y, w, h),
                    "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                }
            )
        return items

    monkeypatch.setattr(image_auto_scene, "run_ocr", sparse_ocr)

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.swin_transformer_architecture"
    assert scene["metadata"]["architecture_template"] == "swin_transformer_sparse"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert not any(str(node.get("id", "")).endswith("_frame") for node in scene["nodes"])
    assert "Patch Mergi\nng" in texts
    assert "Swin\nTransformer\nBlock" in texts
    assert "W-MSA" in texts and "SW-MSA" in texts
    assert len(scene["edges"]) >= 12


def test_mask_res_block_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "mask_res_block.png"
    Image.new("RGB", (1113, 741), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Conv7-64",
            "Batch normalization",
            "ReLU",
            "Max-pooling",
            "Original res-block",
            "Mask res-block",
            "Mask_i",
            "x_i",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.mask_res_block"
    assert scene["metadata"]["architecture_template"] == "mask_res_block"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert "Conv7-64" in texts
    assert "Batch normalization" in texts
    assert "Max-pooling" in texts
    assert "(a) Original res-block" in texts
    assert "(b) Mask res-block" in texts
    assert any(node.get("id") == "right_gate1" and node.get("symbol") == "x" for node in scene["nodes"])
    assert len(scene["edges"]) >= 25


def test_cross_attention_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "cross_attention.png"
    Image.new("RGB", (1368, 438), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "AM-ResNet",
            "Wav2vec 2.0",
            "features",
            "FC",
            "Vw",
            "Kw",
            "Qa",
            "Qw",
            "Ka",
            "Va",
            "Softmax",
            "Concat",
            "norm",
            "Feed forward",
            "Cross-fused features",
            "Fig. 7. The architecture of the cross-attention.",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.cross_attention"
    assert scene["metadata"]["architecture_template"] == "cross_attention"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert sum(1 for node in scene["nodes"] if node.get("type") == "grid_matrix") >= 4
    assert "AM-ResNet\nfeatures" in texts
    assert "Wav2vec 2.0\nfeatures" in texts
    assert "Cross-fused\nfeatures" in texts
    assert "Fig. 7. The architecture of the cross-attention." in texts
    assert len(scene["edges"]) >= 35


def test_attention_mechanism_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "attention_mechanism.png"
    Image.new("RGB", (743, 354), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Attention mechanism",
            "Sigmoid",
            "Weighted vector",
            "Conv1d",
            "High-level features",
            "AM-ResNet features",
            "Fig. 5. The architecture of the attention mechanism.",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])
    node_types = {str(node.get("id")): node.get("type") for node in scene["nodes"]}

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.attention_mechanism"
    assert scene["metadata"]["architecture_template"] == "attention_mechanism"
    assert scene["metadata"]["raster_tile_policy"] == "semantic_template_no_raster_tiles"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert node_types["high_level_features"] == "feature_map_banded"
    assert node_types["weighted_vector"] == "grid_matrix"
    assert node_types["am_resnet_features"] == "feature_map_grid"
    assert node_types["multiply_op"] == "operator_node"
    assert "Attention mechanism" in texts
    assert "Sigmoid" in texts
    assert "Conv1d" in texts
    assert "Weighted vector" in texts
    assert "High-level features" in texts
    assert "AM-ResNet features" in texts
    assert "Fig. 5. The architecture of the attention mechanism." in texts
    assert len(scene["edges"]) == 6


def test_remote_sensing_rsei_workflow_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "remote_sensing_rsei.png"
    Image.new("RGB", (1080, 614), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Images data",
            "Landsat 5 TM",
            "Landsat 8 OLI",
            "JRC Global Surface Water Mapping Layers",
            "Surface Reflectance images",
            "Driver Layer",
            "Terrain",
            "Climate",
            "Soil",
            "Urbanization",
            "Pre-processing",
            "LEDAPS",
            "LaSRC",
            "CFMASK",
            "Mosaic",
            "Extracting",
            "Water Mask",
            "RSEI information extraction by GEE",
            "NDVI",
            "NDSI",
            "WET",
            "LST",
            "Normalization, PCA",
            "Multi-year RSEI maps",
            "RSEI change analysis",
            "PLS-SEM analysis",
            "Global spatial auto-correlation",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])
    node_types = [node.get("type") for node in scene["nodes"]]

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.remote_sensing_rsei_workflow"
    assert scene["metadata"]["architecture_template"] == "remote_sensing_rsei_workflow"
    assert scene["metadata"]["raster_tile_policy"] == "semantic_template_no_raster_tiles"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert node_types.count("dashed_region") >= 8
    assert node_types.count("group_container") >= 1
    assert node_types.count("tensor_stack") >= 2
    assert node_types.count("feature_map_grid") >= 1
    assert node_types.count("grid_matrix") >= 1
    assert node_types.count("polygon_node") >= 10
    assert node_types.count("ellipse_node") >= 10
    assert node_types.count("rounded_process") >= 8
    for label in [
        "Images data",
        "Driver Layer",
        "RSEI information extraction by GEE",
        "NDVI",
        "NDSI",
        "WET",
        "LST",
        "Normalization, PCA",
        "PLS-SEM analysis",
        "Global spatial auto-correlation",
        "RSEI change analysis",
    ]:
        assert label in texts
    assert len(scene["edges"]) >= 34


def test_drought_basin_workflow_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "drought_basin_workflow.png"
    Image.new("RGB", (981, 1417), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Datasets input",
            "Meteorological data",
            "SST data",
            "Nino 3.4 data",
            "River basins data",
            "Drought index SPEI-12",
            "Drought-wet change",
            "PRE trend",
            "Temporal variation",
            "PET trend",
            "Spatial patterns",
            "34 major global river basins",
            "3-D Drought Clustering",
            "Index threshold: -1",
            "Area threshold: 1.56%",
            "Space-time domain :3x3x3",
            "Drought structure",
            "Time n+2",
            "Drought event characteristics",
            "Drought duration",
            "Drought displacements",
            "Drought number",
            "Drought area",
            "Spatiotemporal structure of typical drought event",
            "The Koppen-Geiger climate classification",
            "Influencing factors of drought",
            "Maximum covariance analysis",
            "SST",
            "Drought",
            "ENSO",
            "Spatiotemporal patterns of the MCA2 mode",
            "Identification and contrast of meteorological drought of global river basins",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])
    node_types = [node.get("type") for node in scene["nodes"]]

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.drought_basin_workflow"
    assert scene["metadata"]["architecture_template"] == "drought_basin_workflow"
    assert scene["metadata"]["raster_tile_policy"] == "semantic_template_no_raster_tiles"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert node_types.count("group_container") >= 4
    assert node_types.count("rounded_process") >= 20
    assert node_types.count("grid_matrix") >= 4
    assert node_types.count("polygon_node") >= 12
    assert node_types.count("ellipse_node") >= 4
    for label in [
        "Datasets input",
        "Drought index SPEI-12",
        "Drought-wet change",
        "34 major global river basins",
        "3-D  Drought Clustering",
        "Drought event characteristics",
        "Influencing factors of drought",
        "Maximum covariance analysis",
        "Spatiotemporal patterns of the MCA2 mode",
        "Identification and contrast of meteorological drought of global river basins",
    ]:
        assert label in texts
    assert len(scene["edges"]) >= 45


def test_industry_4_0_sustainability_framework_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "industry_4_0_sustainability_framework.png"
    Image.new("RGB", (981, 561), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Industry4.0",
            "Industry4.0Sustainability",
            "Functions",
            "Sustainable",
            "Manufacturing",
            "Technologies",
            "Artificial intelligence",
            "Mixed Reality",
            "lloT",
            "Blockchain",
            "Digital twins",
            "Robotics",
            "Big data analytics",
            "CPS",
            "Components",
            "Smart customers",
            "Smart distribution",
            "Digital supply networks",
            "Smart shareholders",
            "Smart factory",
            "Smart products",
            "Principles",
            "Virtualization",
            "Vertical integration",
            "Real-time capability",
            "Interoperability",
            "Technical assistance",
            "Decentralization",
            "Horizontal integration",
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
            "Social development",
            "Sustainable economic growth",
            "Renewables",
            "Green manufacturing",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])
    node_types = [node.get("type") for node in scene["nodes"]]

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.industry_4_0_sustainability_framework"
    assert scene["metadata"]["architecture_template"] == "industry_4_0_sustainability_framework"
    assert scene["metadata"]["raster_tile_policy"] == "semantic_template_no_raster_tiles"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert node_types.count("dashed_region") >= 3
    assert node_types.count("group_container") >= 7
    assert node_types.count("rounded_process") >= 28
    assert node_types.count("ellipse_node") >= 12
    assert node_types.count("polygon_node") >= 7
    for label in [
        "Industry 4.0",
        "Technologies",
        "Components",
        "Principles",
        "Business model innovation",
        "Customer-oriented manufacturing",
        "Sustainable value-creation networking",
        "Sustainable\nManufacturing",
        "Social development",
        "Sustainable economic\ngrowth",
        "Renewables",
        "Green manufacturing",
    ]:
        assert label in texts
    assert len(scene["edges"]) >= 20


def test_channel_attention_recalibration_uses_editable_shape_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "channel_attention.png"
    Image.new("RGB", (981, 469), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Original",
            "image",
            "X",
            "U",
            "F_tr",
            "F_sq(.)",
            "F_ex(.,W)",
            "F_scale(.,.)",
            "1x1xC",
            "1X1XC",
            "H",
            "C",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])
    node_types = [node.get("type") for node in scene["nodes"]]

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.channel_attention_recalibration"
    assert scene["metadata"]["architecture_template"] == "channel_attention_recalibration"
    assert scene["metadata"]["raster_tile_policy"] == "semantic_template_no_raster_tiles"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert node_types.count("cuboid_node") >= 4
    assert node_types.count("tensor_stack") >= 2
    assert node_types.count("feature_vector_stack") >= 4
    assert node_types.count("feature_map_banded") >= 2
    assert any(node.get("type") == "math_text" and "F_scale" in str(node.get("text", "")) for node in scene["nodes"])
    assert "Original" in texts
    assert "image" in texts
    assert "X~" in texts
    assert len(scene["edges"]) >= 14


def test_deformable_transformer_encoder_decoder_uses_editable_template(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "deformable_transformer.png"
    Image.new("RGB", (981, 524), "white").save(source)
    monkeypatch.setattr(
        image_auto_scene,
        "run_ocr",
        lambda _path: fake_ocr_items([
            "Encoder",
            "Decoder",
            "Multi-Head Deformable",
            "Self-Attention",
            "Cross-Attention",
            "Add & Norm",
            "BC-FFN",
            "GN",
            "GELU",
            "Location-guided queries",
            "Feature Grids",
            "Restore",
            "T3-T5",
            "Flatten",
        ]),
    )

    scene = image_auto_scene.build_scene(source, allow_raster_tiles=False, reconstruction_mode="standard")
    texts = "\n".join(str(node.get("text", "")) for node in scene["nodes"])
    node_types = [node.get("type") for node in scene["nodes"]]

    assert scene["metadata"]["created_by"] == "fig4visio.image_auto_scene.deformable_transformer_encoder_decoder"
    assert scene["metadata"]["architecture_template"] == "deformable_transformer_encoder_decoder"
    assert scene["metadata"]["raster_tile_policy"] == "semantic_template_no_raster_tiles"
    assert scene["assets"] == []
    assert all(node.get("type") != "image_tile" for node in scene["nodes"])
    assert node_types.count("group_container") >= 4
    assert all(
        node.get("style", {}).get("fill") == "#F0F0F0"
        for node in scene["nodes"]
        if node.get("type") == "group_container"
    )
    assert node_types.count("grid_matrix") >= 2
    assert node_types.count("tensor_stack") >= 2
    assert node_types.count("feature_vector_stack") >= 2
    assert node_types.count("operator_node") >= 8
    assert "Encoder" in texts
    assert "Decoder" in texts
    assert "BC-FFN" in texts
    assert "Location-guided queries" in texts
    assert "Feature" in texts and "Grids" in texts
    assert len(scene["edges"]) >= 35


def test_self_check_rejects_missing_main_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "source_swin.png"
    bad = tmp_path / "bad_swin.png"
    draw_swin_like_line_art(source)
    draw_swin_like_line_art(bad, omit_left_pipeline=True)

    identical = self_check.compare_images(source, source)
    report = self_check.compare_images(source, bad)

    assert identical["passed"] is True
    assert identical["failed_rules"] == []
    assert report["passed"] is False
    assert report["failed_rules"]
    assert report["metrics"]["grid_density_similarity"] < report["rules"]["min_grid_density_similarity"] or report["score"] < report["threshold"]


def test_gui_self_check_summary_reports_gate_failure_after_score_passes() -> None:
    report = {
        "passed": False,
        "score": 0.417,
        "threshold": 0.38,
        "failed_rules": [
            {
                "rule": "grid_density_similarity",
                "metric": "grid_density_similarity",
                "value": 0.0,
                "required": 0.32,
            }
        ],
    }

    summary = gui_app.format_self_check_failure_summary(report, [{}, {}, {}])

    assert "已达到阈值" in summary
    assert "网格密度分布" in summary
    assert "低于阈值" not in summary


def test_rounded_orthogonal_points_rounds_only_the_corner() -> None:
    points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    rounded = rounded_orthogonal_points(points, corner_radius=0.5, samples_per_corner=4)

    assert rounded[0] == points[0]
    assert rounded[-1] == points[-1]
    assert (2.0, 0.0) not in rounded
    assert any(abs(x - 1.5) < 1e-9 and abs(y - 0.0) < 1e-9 for x, y in rounded)
    assert any(abs(x - 2.0) < 1e-9 and abs(y - 0.5) < 1e-9 for x, y in rounded)
    assert all(
        0.0 <= x <= 2.0 and 0.0 <= y <= 2.0
        for x, y in rounded
    )


def test_pixel_corner_radius_is_scaled_to_inches() -> None:
    scene = {
        "version": "0.1",
        "page": {"width": 1000, "height": 500, "units": "px", "target_width_in": 10},
        "nodes": [
            {"id": "a", "type": "rounded_process", "x": 100, "y": 100, "w": 100, "h": 50, "text": "A"},
            {"id": "b", "type": "rounded_process", "x": 800, "y": 300, "w": 100, "h": 50, "text": "B"},
        ],
        "edges": [
            {
                "id": "a_to_b",
                "type": "rounded_orthogonal_connector",
                "from": "a:right@0.50",
                "points": [[400, 125], [400, 325]],
                "to": "b:left@0.50",
                "route": "rounded_orthogonal",
                "corner_radius_px": 12,
            }
        ],
        "assets": [],
    }

    normalized = normalize_scene_coordinates(scene)
    edge = normalized["edges"][0]
    assert edge["corner_radius_in"] == 0.12


def test_rounded_orthogonal_connector_validates_and_routes() -> None:
    scene = {
        "version": "0.1",
        "page": {"width": 8, "height": 4.5, "units": "in"},
        "nodes": [
            {"id": "a", "type": "rounded_process", "x": 1, "y": 1, "w": 1, "h": 0.5, "text": "A"},
            {"id": "b", "type": "rounded_process", "x": 5, "y": 2, "w": 1, "h": 0.5, "text": "B"},
        ],
        "edges": [
            {
                "id": "a_to_b",
                "type": "rounded_orthogonal_connector",
                "from": "a:right@0.50",
                "points": [[3, 1.25], [3, 2.25]],
                "to": "b:left@0.50",
                "route": "rounded_orthogonal",
                "corner_radius_in": 0.12,
            }
        ],
        "assets": [],
    }
    scene_path = Path.cwd() / "__tmp_rounded_orthogonal.scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    try:
        result = run_script("scene_validate.py", str(scene_path))
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        scene_path.unlink(missing_ok=True)

    component_map = load_component_map()
    edge = scene["edges"][0]
    style = edge_style(edge, component_map, {})
    nodes = {node["id"]: node for node in scene["nodes"]}
    route = edge_route_points(edge, style, nodes)
    assert route == [(2.0, 1.25), (3.0, 1.25), (3.0, 2.25), (5.0, 2.25)]


def test_rounded_orthogonal_example_passes_strict_contract() -> None:
    example = ROOT / "templates" / "examples" / "rounded_orthogonal_connector.scene.json"
    result = run_script("scene_validate.py", str(example), "--strict")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Scene is valid" in result.stdout


def test_review_bundle_and_checklist_gate(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    replica = tmp_path / "replica.png"
    scene = tmp_path / "scene.json"
    review_dir = tmp_path / "review"

    Image.new("RGB", (320, 180), "white").save(original)
    Image.new("RGB", (320, 180), (245, 248, 255)).save(replica)
    scene.write_text(
        json.dumps(
            {
                "version": "0.1",
                "page": {"width": 8, "height": 4.5, "units": "in"},
                "nodes": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    bundle = run_script(
        "make_review_assets.py",
        "--original",
        str(original),
        "--replica",
        str(replica),
        "--scene",
        str(scene),
        "--id",
        "smoke",
        "--round",
        "1",
        "--write-review-bundle",
        "--output-dir",
        str(review_dir),
    )
    assert bundle.returncode == 0, bundle.stdout + bundle.stderr

    findings = review_dir / "smoke_review_findings.json"
    manifest = review_dir / "smoke_review_manifest.json"
    findings.write_text(
        json.dumps(
            {
                "figure_id": "smoke",
                "round": 1,
                "overall_verdict": "needs_rebuild",
                "rebuild_required": True,
                "topology_checklist": [
                    {
                        "id": "T001",
                        "focus_region": "main",
                        "source_fact": "A visible arrow should connect A to B.",
                        "replica_status": "The arrow is missing.",
                        "status": "fail",
                        "certainty": "certain",
                    }
                ],
                "visual_checklist": [
                    {
                        "id": "V001",
                        "focus_region": "main",
                        "source_expectation": "The two boxes are horizontally aligned.",
                        "replica_status": "The boxes are vertically offset.",
                        "status": "fail",
                        "certainty": "certain",
                    }
                ],
                "findings": [
                    {
                        "id": "F001",
                        "severity": "blocking",
                        "summary": "Main arrow is missing",
                        "visible_diff": "The source has a connector but the replica does not.",
                        "source_appearance": "Two boxes joined by one arrow.",
                        "replica_appearance": "Two unconnected boxes.",
                        "impact_on_fidelity": "The diagram topology is wrong.",
                        "focus_regions": ["main"],
                        "checklist_refs": ["T001", "V001"],
                        "expected_visible_change": "Restore the connector from A to B.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    gate = run_script(
        "review_checklist_gate.py",
        str(findings),
        "--manifest",
        str(manifest),
        "--require-failed-refs",
    )
    assert gate.returncode == 0, gate.stdout + gate.stderr


def test_arrow_plan_gate_accepts_bound_horizontal_arrow(tmp_path: Path) -> None:
    scene = tmp_path / "arrow_plan_ok.scene.json"
    scene.write_text(
        json.dumps(
            {
                "version": "0.1",
                "metadata": {
                    "arrow_plan": [
                        {
                            "id": "A001",
                            "from": "left box right boundary",
                            "from_visual_object": "left box",
                            "to": "right box left boundary",
                            "to_visual_object": "right box",
                            "from_anchor": "right@0.50",
                            "from_anchor_description": "right edge midpoint",
                            "to_anchor": "left@0.50",
                            "to_anchor_description": "left edge midpoint",
                            "semantic_intent": "data_flow",
                            "route_shape": "straight_horizontal",
                            "line_style": "solid",
                            "direction": "left_to_right",
                            "arrowhead": "end",
                            "must_be_axis_aligned": True,
                            "source_bbox_px": [100, 100, 260, 120],
                            "must_not_cross": ["a", "b"],
                            "relative_position_facts": ["left box is left of right box", "arrow is horizontal"],
                            "certainty": "certain",
                        }
                    ]
                },
                "page": {"width": 8, "height": 4.5, "units": "in"},
                "nodes": [
                    {"id": "a", "type": "rounded_process", "x": 1, "y": 2, "w": 1, "h": 0.5, "text": "A"},
                    {"id": "b", "type": "rounded_process", "x": 4, "y": 2, "w": 1, "h": 0.5, "text": "B"},
                ],
                "edges": [
                    {
                        "id": "a_to_b",
                        "type": "lane_arrow",
                        "arrow_plan_id": "A001",
                        "from": "a:right@0.50",
                        "to": "b:left@0.50",
                        "route": "horizontal",
                    }
                ],
                "assets": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_script("scene_validate.py", str(scene), "--strict")
    assert result.returncode == 0, result.stdout + result.stderr


def test_arrow_plan_gate_rejects_diagonal_horizontal_arrow(tmp_path: Path) -> None:
    scene = tmp_path / "arrow_plan_bad.scene.json"
    scene.write_text(
        json.dumps(
            {
                "version": "0.1",
                "metadata": {
                    "arrow_plan": [
                        {
                            "id": "A001",
                            "from": "left box right boundary",
                            "from_visual_object": "left box",
                            "to": "right box left boundary",
                            "to_visual_object": "right box",
                            "from_anchor_description": "right edge midpoint",
                            "to_anchor_description": "left edge midpoint",
                            "semantic_intent": "data_flow",
                            "route_shape": "straight_horizontal",
                            "line_style": "solid",
                            "direction": "left_to_right",
                            "arrowhead": "end",
                            "must_be_axis_aligned": True,
                            "source_bbox_px": [100, 100, 260, 120],
                            "must_not_cross": ["a", "b"],
                            "relative_position_facts": ["left box is left of right box", "arrow is horizontal"],
                            "certainty": "certain",
                        }
                    ]
                },
                "page": {"width": 8, "height": 4.5, "units": "in"},
                "nodes": [
                    {"id": "a", "type": "rounded_process", "x": 1, "y": 1, "w": 1, "h": 0.5, "text": "A"},
                    {"id": "b", "type": "rounded_process", "x": 4, "y": 2, "w": 1, "h": 0.5, "text": "B"},
                ],
                "edges": [
                    {
                        "id": "a_to_b",
                        "type": "lane_arrow",
                        "arrow_plan_id": "A001",
                        "from": "a:right@0.50",
                        "to": "b:left@0.50",
                        "route": "straight",
                    }
                ],
                "assets": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_script("scene_validate.py", str(scene), "--strict")
    assert result.returncode != 0
    assert "expects a horizontal arrow" in (result.stdout + result.stderr)
