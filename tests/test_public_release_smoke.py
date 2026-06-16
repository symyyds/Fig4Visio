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


def test_self_check_rejects_missing_main_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "source_swin.png"
    bad = tmp_path / "bad_swin.png"
    draw_swin_like_line_art(source)
    draw_swin_like_line_art(bad, omit_left_pipeline=True)

    identical = self_check.compare_images(source, source)
    report = self_check.compare_images(source, bad)

    assert identical["passed"] is True
    assert report["passed"] is False
    assert report["metrics"]["grid_density_similarity"] < report["rules"]["min_grid_density_similarity"] or report["score"] < report["threshold"]


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
