from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


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
