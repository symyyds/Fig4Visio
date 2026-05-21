---
name: visiomaster
description: Windows-first Visio diagram reconstruction workflow for flowcharts, architecture diagrams, and paper-style module figures. Reuses ppt-master style analysis and composition discipline on the front half, but outputs editable Visio .vsdx plus exported .svg and .png through a scene.json to Visio pipeline. Use when the user wants a diagram recreated as editable Visio shapes instead of a pasted screenshot or PPT-only result.
---

# Visiomaster

## Overview

`visiomaster` is a standalone skill for rebuilding diagram images into editable Visio deliverables.

It is optimized for:
- flowcharts
- product or system architecture diagrams
- paper-style module/framework figures
- box-arrow process diagrams that should remain editable

It is not the right tool for:
- posters
- UI screenshots
- decorative layouts
- image-heavy slides where the main value is visual styling rather than structured diagram semantics

## Core Positioning

Use `ppt-master` ideas on the front half:
- source collection
- style extraction
- layout discipline
- image understanding
- visual polishing standards

Do **not** reuse `ppt-master`'s raw `SVG -> PPTX` output path for Visio.

For Visio, the stable path is:

`image -> scene.json -> validate -> Visio COM render -> .vsdx/.svg/.png`

The key rule is simple:
- main structure should be redrawn as editable nodes, labels, and connectors
- small thumbnails or texture snippets may remain raster only when redrawing them is not worth the loss in speed
- never solve a reconstruction request by pasting the whole original image unless the user explicitly asks for贴图

## Environment

This skill is Windows-first and expects:
- local Microsoft Visio desktop installed
- Python with `pywin32`

Use the active environment's Python interpreter. The examples below use `python`; replace it with a project-specific interpreter path when needed.

## Workflow

### 1. Confirm scope

First classify the source request:
- editable flowchart recreation
- architecture/module diagram recreation
- paper figure cleanup/redraw
- image-assisted redraw with a few raster sub-assets allowed

If the diagram is mostly boxes, arrows, labels, and containers, stay in `visiomaster`.

For wide or dense figures, do not start by authoring the whole page in one pass. If the source has many modules, many arrows, tiny labels, or a very wide canvas, first create a region plan:
- visible regions become `group_container`
- invisible logical work areas become `audit_region`
- every meaningful node gets `container_id`
- each region should usually stay under 12-18 visible nodes before whole-page assembly
- shared typography and arrow styles must be fixed before region scenes are merged

### 2. Build or refine `scene.json`

`scene.json` is the contract between the visual analysis step and Visio rendering.

When authoring or editing it:
- read `references/scene-schema.md`
- use `templates/visio_components.json` as the supported component vocabulary
- use `templates/style_profiles.json` to select `paper_white` or `clean_white`
- read `references/visio-component-map.md` if you need mapping guidance
- for exact replicas, author coordinates in source pixels when possible and let the renderer scale them to inches

Starter generation:

```powershell
python ${SKILL_DIR}\scripts\image_to_scene.py --image <source.png> --output <scene.json>
```

For exact large-figure reconstruction, start in source pixels and record the region strategy:

```powershell
python ${SKILL_DIR}\scripts\image_to_scene.py --image <source.png> --pixel-page --region-strategy region_first --output <scene.json>
```

If the layout is close to a standard process flow, you can seed from the built-in example:

```powershell
python ${SKILL_DIR}\scripts\image_to_scene.py --image <source.png> --template basic-flow --output <scene.json>
```

For GAN/TFR training-cycle figures, AI-generated paper diagrams, or images with Real/Generated TFR panels, do not start from a blank scene. Seed the first pass from the canonical module template:

```powershell
python ${SKILL_DIR}\scripts\image_to_scene.py --image <source.png> --template gan-tfr --output <scene.json>
```

Before the first Visio render of a hand-authored or legacy GAN/TFR scene, run the deterministic recipe pass:

```powershell
python ${SKILL_DIR}\scripts\scene_autofix.py <scene.json> --recipe gan-tfr --output <fixed.scene.json>
```

This recipe upgrades fragile local grammar before visual tuning: split Real/Generated boxes become `tfr_panel`, empty dashed loss frames become `loss_region`, raw `L_adv`/`L_rec` text becomes `math_text`, detached/broken outer loops become smooth `loop_arrow`, reversed GAN arrows are corrected, and crowded backprop arrows are bundled.

`scene_to_visio.py` also runs this GAN/TFR autofix once by default before the rebuild gate. Treat the written `<basename>.autofixed.scene.json` as the scene to inspect when the renderer reports pre-render changes. Use `--no-autofix` only when you intentionally want to debug the raw scene.

### 3. Validate scene data

Before touching Visio, validate structure and references:

```powershell
python ${SKILL_DIR}\scripts\scene_validate.py <scene.json>
```

If validation fails, fix the scene first. Do not guess around broken ids or unsupported types inside the renderer.

For large or complex figures, run a complexity preflight before full rendering:

```powershell
python ${SKILL_DIR}\scripts\scene_complexity.py <scene.json>
```

Use the complexity report to catch the large-image failure modes before Visio render: too few regions, uncovered nodes, over-dense modules, inconsistent font scale, text-fit risks, and likely overlaps.

For exact replicas, run a typography preflight when the source uses more than one visible font style:

```powershell
python ${SKILL_DIR}\scripts\font_inventory.py --check "Times New Roman" --check "Cambria Math" --check "Calibri" --check "Microsoft YaHei UI"
```

Do not treat all labels as one font. Classify visible text by role before rendering:
- paper serif labels: `font_role: "paper_serif"` with Times/Cambria-like candidates
- UI/product labels: `font_role: "ui_sans"` with Calibri/Arial/Segoe-like candidates
- formulas/operators: `font_role: "math"` or `symbol_font_role: "math"`
- Chinese labels: `font_role: "cjk_sans"` or `cjk_serif`

When the source font is known or strongly inferred, store `source_font_family` in the node style. If that font is installed but the effective render font differs, `scene_audit.py` reports it as a rebuild issue.

For complex paper figures with many modules, also generate a module-level audit report:

```powershell
python ${SKILL_DIR}\scripts\scene_audit.py <scene.json>
```

Use the audit report to review every `group_container` as a separate region: child count, labels, colors, internal arrows, incoming arrows, outgoing arrows, and whether cross-module arrows start from a boundary or from an internal component. Treat unchecked audit items as real defects before final export.

For exact replicas, run the rebuild gate after each rendered iteration:

```powershell
python ${SKILL_DIR}\scripts\scene_audit.py <scene.json> --fail-on-rebuild
```

If the audit prints `[REBUILD]` items, stop coordinate nudging. Rebuild that local subsystem with the correct semantic component (`loop_arrow`, `dashed_region`, `dashed_feedback_path`, `boundary_port`, etc.) before doing any more visual polishing.

For exact replica work, validation passing is necessary but not sufficient. Render a PNG and compare it with the source for:
- page aspect ratio
- container bounds
- local topology
- distinctive shapes
- connector grammar
- feature-map coloring

### 4. Render into Visio

Render the scene into a Visio drawing and export deliverables:

```powershell
python ${SKILL_DIR}\scripts\scene_to_visio.py <scene.json> --output-dir <exports>
```

Default outputs:
- `.vsdx`
- `.svg`
- `.png`

Use `--style-profile clean_white` when the user wants a polished white product/process style. Use the default `paper_white` for paper figures and academic module diagrams.

Read `references/visio-export-flow.md` when debugging Visio automation or export behavior.

After rendering complex replicas, open the exported PNG and compare it module-by-module against the source. Do not rely on whole-image visual similarity; small topology errors often hide inside large figures.

## Component Strategy

Version 1 intentionally uses a **small controlled vocabulary** instead of trying to expose all Visio masters.

Supported node families:
- `page_background`
- `process_box`
- `rounded_process`
- `stacked_process`
- `stacked_token`
- `notched_block`
- `feature_map_banded`
- `feature_map_grid`
- `merge_bus`
- `decision_diamond`
- `terminator`
- `group_container`
- `dashed_region`
- `loss_region`
- `audit_region`
- `text_pill`
- `ellipse_node`
- `polygon_node`
- `trapezoid_node`
- `cuboid_node`
- `modality_spine`
- `math_vector`
- `math_text`
- `tfr_panel`
- `operator_node`
- `boundary_port`
- `boundary_fanout`
- `wave_signal`
- `classifier_head`
- `text_block`
- `grid_matrix`
- `bracket`
- `junction_point`
- `image_tile`
- `legend_block`

Supported edge families:
- `arrow_connector`
- `dynamic_connector`
- `lane_arrow`
- `curved_arrow`
- `loop_arrow`
- `dashed_feedback_path`
- `line_segment`
- `join_connector`
- `fork_connector`
- `boundary_arrow`
- `residual_connector`
- `residual_loop`

Why this matters:
- it keeps `scene.json` stable
- it avoids binding the whole system to localized Visio stencil names too early
- it lets us start with primitive geometry rendering, then add real stencil/master mapping later without breaking the scene schema

## Execution Rules

1. Prefer editable reconstruction over screenshot embedding.
2. Recreate hierarchy first: containers, major nodes, main connectors, then secondary labels.
3. Preserve the source image's information design before chasing decorative detail.
4. Keep coordinates in `scene.json` in top-left page space; let the renderer convert to Visio coordinates.
5. If a source figure contains one non-essential photographic or map tile, isolate that asset instead of rasterizing the full page.
6. When a shape is ambiguous, fall back to the nearest supported component and note the approximation.
7. For arrows, use `route` and explicit `points`; do not rely on diagonal lines unless the source really uses diagonals.
8. For rotated paper labels, use `text_block` with `angle_deg` instead of rotating text inside a process shape.
9. For convolution kernels, receptive fields, masks, and other regular cell diagrams, use `grid_matrix`; do not manually author each square.
10. For modality grouping marks such as `]`, `[`, `U`, and inverted `U`, use `bracket`; do not fake them with ultra-thin process boxes.
11. For 2-to-1, 3-to-1, or 1-to-many arrows, place a tiny `junction_point` at the merge/fan location, connect sources to the junction, then connect the junction to the destination.
12. Do not connect arrows directly to `group_container`. Containers frame regions only; use a nearby `junction_point` or explicit node on the border when a callout line is needed.
13. For brackets with a middle merge arm, set `tick_positions: [0, 0.5, 1]`; a plain two-arm bracket is not enough for modality merge symbols.
14. For cross-container flow, split the edge through `junction_point` nodes with `role: boundary_anchor`; set `allow_cross_container: true` only on the short bridge between anchors.
15. For dense mini-module diagrams, keep all connectors axis-aligned with `hv`, `vh`, or explicit aligned points. A connector must not cross through a non-endpoint node.
16. Run `scene_validate.py` after authoring. Treat route-quality warnings as defects, not cosmetic suggestions, before rendering through Visio.
17. For repeated offset feature blocks, use `stacked_process`; do not author each visible layer as an independent process node unless each layer is a real semantic node.
18. For module interiors such as AM-ResNet, encode intended alignment with `align_to_container` or `align_group`; do not rely on eyeballed y values.
19. For source lines that are only visual stubs or bus segments, use `line_segment` with `from_point`/`to_point`. Do not force those lines to terminate on a component just because Visio can connect to shapes.
20. For 1:1 or exact replica requests, set `metadata.fidelity: "exact"` and include `metadata.source_image` or `metadata.source_aspect_ratio`. Do not deliver only because validation passes; compare the rendered PNG against the source and revise until proportions, module positions, special shapes, and connector semantics match.
21. Do not invent random feature maps, generic CNN rectangles, or approximate classifier wiring when the source has distinctive visual encodings. Encode those visual encodings explicitly with `grid_matrix`, `line_segment`, point endpoints, or isolated small raster assets when editability is less important than faithful local appearance.
22. Prefer pixel coordinate authoring for exact replicas: set `page.units: "px"`, `page.width`, `page.height`, and `page.target_width_in`; the renderer will normalize nodes and edge points to inches.
23. Use `notched_block` for CNN or cutout modules, `feature_map_banded` for simple striped/columnar feature maps, `feature_map_grid` for AM-ResNet/heatmap-like feature maps with colored rows and shaded columns, `merge_bus` for visible bus/spine merges, and `residual_connector` for residual or skip loops.
24. Use `scripts/enumerate_visio_masters.py` only to research local Visio master names. Do not hard-code a large master catalog into scenes; local Office language and stencil availability vary.
25. For paper module figures, prefer semantic primitives over generic boxes: use `operator_node` for +/x/tensor operators, `boundary_port` for frame entry/exit anchors, `boundary_fanout` for arrows emitted from a dashed frame boundary, `wave_signal` for waveform inputs, `classifier_head` for AvgPool/Linear blocks, `join_connector` for source-to-merge legs, and `fork_connector` for fan-out branches.
26. When an arrow must meet a dashed frame edge, do not connect to the frame. Place a small `boundary_port` on the frame boundary, connect internal flow to that port, then use a short bridge segment or connector to the next module.
27. Do not use `classifier_head` internal fan-out when the source shows output arrows starting on the container boundary. Set `classifier_head.output_mode: "boundary"` or omit `fanout_count`, then draw those branches with `boundary_fanout`.
28. For compact internal arrows such as `AvgPool -> Linear`, use `classifier_head.orientation: "vertical"` with `internal_arrow_size: "tiny"`, or set `arrow_size: "small"` on a normal edge. A default Visio arrowhead can consume very short line segments and appear as a head-only triangle.
29. For operator symbols, use one `operator_node` with `symbol` instead of an `ellipse_node` plus separate text. The renderer centers `+`, `×`, and `⊗` inside the circle.
30. For arrows from a frame boundary to a target component at a different vertical position, use side-ratio endpoints such as `feature_map:left@0.58` to preserve a horizontal lane. Do not accept a diagonal center-to-center edge unless the source truly shows a diagonal.
31. When the source shows the dashed frame itself exporting to the next module, use `boundary_arrow` from a `boundary_port` and do not draw an internal `line_segment` from the last component to the frame. Component-to-frame-to-component long lines are usually wrong unless visibly present in the source.
32. For ordinary 1-to-1 arrows, prefer `route: "horizontal"` or `route: "vertical"` when the source line is straight. Reserve diagonal arrows for fan-in/fan-out, callouts, or explicit source diagonals.
33. For complex figures, run `scripts/scene_audit.py` and review the generated checklist by module before delivery. The goal is to catch plausible-looking but wrong details: missing dashed frames, wrong child counts, hidden cross-frame arrows, misplaced operators, and incorrect boundary outputs.
34. If the source has no visible dashed module frames, add invisible `audit_region` nodes around logical areas such as residual block, attention block, classifier head, and feature extraction chain. These regions do not render but make `scene_audit.py` review the figure module-by-module.
35. For multimodal paper pipelines with RGB/IR/SAR inputs, preserve each modality lane as a lane: `image_tile` input, `text_pill` or `process_box` availability flag, `modality_spine` shared vertical bus, and explicit horizontal connectors into the fusion module.
36. Use `modality_spine` for vertical gray shared-response or availability-mask bars with repeated `P_RGB`/`P_IR`/`P_SAR` side ports. Do not model the bar as a plain rectangle plus unrelated tiny boxes unless each port has separate topology.
37. Use `cuboid_node` for editable 3D paper blocks such as modality-related impact factors, stacked feature tensors, and small blue/orange depth blocks. Use `feature_map_grid` only when the visible face is a grid or heatmap.
38. Use `trapezoid_node` for quality heads, extractor heads, aggregation modules, and other wedge/trapezoid paper blocks. Set `orientation` and `pointed` so the narrow side or tip matches the source direction.
39. Use `polygon_node` only when the shape cannot be expressed by `trapezoid_node`, `cuboid_node`, `notched_block`, or another semantic primitive. Include normalized points when authoring in source-pixel scenes.
40. For formula/vector annotations such as `q = [q_RGB, q_IR, q_SAR]^T`, use `math_vector` instead of a plain multi-line `text_block`. This keeps brackets, entries, and the optional prefix aligned as editable shapes/text. Do not make formula labels connector endpoints.
41. For wide multi-panel figures, create `audit_region` nodes for every titled panel even when the visible frame is drawn with `group_container`; this lets `scene_audit.py` review inputs, outputs, and cross-panel arrows panel by panel.
42. For large source images, use `metadata.region_strategy: "region_first"` or `"tiled_subscenes"` before rendering. If the figure is too complex to reason about at full-page scale, author each region as a local subscene, validate it, then copy the region nodes into the full scene with the same style tokens.
43. Do not let every region invent its own font sizes. Define a small scale and reuse it: frame titles, body labels, small labels, operator symbols, formula labels, and edge labels. If `scene_validate.py` reports same-type font spread, normalize the style before rendering.
44. When reconstructing from cropped subregions, keep crop-local coordinates only during analysis. Convert to the full-page pixel coordinate system before final assembly so arrows and labels do not drift at seams.
45. Use `scene_complexity.py` before full-page Visio output whenever the scene has roughly 30+ visible nodes, 35+ edges, or a very wide aspect ratio. Treat text-fit, overlap, uncovered-node, and dense-region warnings as actionable defects.
46. A region with more than 18 visible nodes is usually too dense for reliable one-pass reconstruction. Split it into smaller invisible `audit_region` areas such as input stack, feature extractor, fusion, classifier, and output head.
47. For large figures, never fix alignment only by nudging text boxes after the whole render. First enforce region-local alignment with `align_group`, `align_to_container`, explicit `container_id`, and side-ratio endpoints; then tune final positions.
48. In whole-page assembly, use cross-region edges only for true inter-module flow. Keep internal arrows inside their source region and use boundary ports or junction anchors when a connector leaves a region.
49. For short paper-flow lanes between nearby blocks, especially cube/feature blocks into extractor or GAP/GMP blocks, use `lane_arrow` with `route: "horizontal"` or `"vertical"`. Do not use `arrow_connector` with `route: "straight"` and `allow_diagonal: true` to hide small endpoint y/x mismatches.
50. If a generated diagram has arrows that look slightly tilted, inspect the scene for `straight` edges whose endpoints are almost but not exactly aligned. Convert them to `lane_arrow`, force the route axis, or align `from_point`/`to_point` exactly.
51. For paper wedges such as `Quality Head`, `Environment Response extractor`, and `Aggregation Quality-aware`, prefer editable `trapezoid_node`/`polygon_node` unless the user explicitly accepts a small raster tile for speed. Raster tiles should be recorded as a fidelity/speed tradeoff.
52. For GAN/training-cycle figures with a large outer curved arrow, use one `loop_arrow` or `curved_arrow` with sampled `points` and an explicit `end_tangent_point` near the arrowhead. Do not split the loop into several `line_segment` arcs plus detached short arrowheads; that causes visible breaks and wrong tangent direction.
53. For visible dashed annotation boxes such as `Forward Reconstruction -> Discriminator Evaluation`, use `dashed_region` with separate `text_block` labels. Do not use an empty dashed `process_box` as a fake frame.
54. For dashed reconstruction/adversarial/backpropagation paths, use `dashed_feedback_path` with explicit orthogonal points. Do not set `allow_diagonal: true` on loss/backprop arrows just to suppress warnings.
55. When a dashed feedback path leaves a visible dashed region, encode the exact crossing intentionally with `allow_cross_container: true` and explicit points. If the source arrow starts at the frame edge, add a `boundary_port` instead of connecting from the region center.
56. If `scene_audit.py --fail-on-rebuild` reports `[REBUILD]`, do not spend another iteration moving boxes or text. Replace the wrong local grammar first; only after the rebuild gate passes should you tune positions.
57. If the same `[REBUILD]` item appears after one attempted fix, discard that local subsystem and redraw it from a minimal local scene. This is mandatory for outer loops, dashed feedback paths, and dashed annotation frames.
58. For Visio PNG export crop control, use `page_background` as the bottom node. Do not use a fake `process_box` background, because it pollutes route intersection checks and audit output.
59. For GAN/TFR loop figures, the minimum delivery gate is: no passive ellipse used as the training loop, no detached loop arrowheads, no empty dashed process boxes, no dashed/loss/backprop route drawn as a plain `arrow_connector`, and no dashed feedback path crossing text labels.
60. In GAN/TFR diagrams, the generated/reconstructed TFR sample feeds into the Discriminator. If an edge runs `Discriminator -> Generated`, treat it as reversed unless the source explicitly marks it as discriminator output.
61. Outer update loops should use `loop_arrow` with `curve_mode: "smooth"`, enough sampled points, `semantic_role: "outer_update_loop"`, and a `label_id`/`loop_label_id` tied to the update label. Otherwise the loop reads like a decorative border.
62. Keep dashed evaluation/loss frames visually clean. A `dashed_feedback_path` should leave a `dashed_region`/`loss_region` from a boundary point or `boundary_port`; do not draw extra horizontal/vertical stubs through the region interior.
63. For bottom GAN loss/backprop systems with multiple vertical dashed arrows into the discriminator, use a shared `merge_bus`/`junction_point` and `bundle_id` rather than several unrelated vertical arrows.
64. For loss formulas such as `L_adv` and `L_rec`, use `math_text` instead of raw underscore `text_block` strings. Raw underscores are a rebuild defect in exact paper-figure replicas.
65. For Real/Generated TFR panels, use `tfr_panel` as the first-pass component. Do not split the panel into a rounded box, title labels, grid, input label, and a separate internal arrow unless there is a source-specific reason.
66. When a render looks globally close but local details do not improve after one pass, run `scene_audit.py --fail-on-rebuild` and fix every `[REBUILD]` item before visual tuning.
67. For GAN/TFR source images, start with `--template gan-tfr` whenever the topology matches. This is a generation-first capability, not a post-render cleanup step.
68. Before rendering a legacy or hand-authored GAN/TFR scene, run `scene_autofix.py --recipe gan-tfr` once. If the recipe rewrites local grammar, validate and audit the fixed scene instead of continuing from the old file. The renderer also runs this pass by default for GAN/TFR scenes and writes `<basename>.autofixed.scene.json`; inspect that file when debugging first-pass generation.
69. Use `loss_region` for the dashed adversarial/evaluation area when the title and formulas belong to one local subsystem. Use `dashed_region` only when the frame has no formula semantics or the source requires separate child nodes.
70. When a `loss_region` and its target block overlap horizontally, route feedback as short vertical boundary stubs into `target:top@ratio` or `target:bottom@ratio`. Do not connect from loss-frame corners to `target:left/right`; that usually creates the false "dashed box plus extra arrow" artifact.
71. If the outer loop arrowhead still looks kinked after smoothing, change the semantic geometry (`end_tangent_point`, sampled points, or local loop subsystem) before nudging unrelated nodes. The arrowhead tangent is part of the component grammar, not a final polish detail.
72. Normalize GAN/TFR loss text before rendering. `Ladv`, `Lrec`, `L adv`, and `L rec` should be converted to `L_adv` / `L_rec` or explicit `math_text` fragments before export.
73. For `loss_region` titles, prefer `title_position: "inside"` or a header-cutout layout. Do not let the dashed frame cross a long title; split the title line or widen the region first.
74. For exact or GAN/TFR renders, do not call `scene_to_visio.py` as a blind final step. Run the rebuild gate first, or let the renderer's built-in gate stop the export when `[REBUILD]` items remain.
75. If a dashed feedback path looks like a tiny isolated arrow fragment, treat that as a semantic-route failure, not a cosmetic issue. Rebuild it as one `dashed_feedback_path` or a bus/port route instead of preserving the fragment.
76. Do a typography pass for exact replicas. Identify whether the source uses serif, sans, math, mono, CJK, or mixed typography before finalizing nodes; do not let every text node inherit a generic default by accident.
77. Use `font_family_candidates` and `font_role` instead of a single fragile font name when exact family matching is uncertain. The renderer resolves to the first installed close match.
78. If the source font is known, set `source_font_family` as well as `font_family`/`font_family_candidates`. This lets audit catch the case where the font exists locally but the scene still renders with the wrong family.
79. Run `scripts/font_inventory.py` when a figure appears to use Calibri/Aptos/Arial/Times/Cambria/Chinese fonts. A missing source font should lead to a candidate list; an installed source font should be used directly.
80. Use math-capable fonts for operators and formula fragments: `Cambria Math`, `Cambria`, or `Times New Roman` depending on source style. Do not render `+`, `×`, `⊗`, or subscript formulas with a random UI font unless the source visibly does.
81. For Chinese or mixed Chinese/English diagrams, use `font_role: "cjk_sans"` or `cjk_serif`; otherwise Visio may silently substitute glyphs and shift text metrics.
82. Review the `Typography Review` section in `scene_audit.py` output before coordinate polishing. Font fallbacks can change text width and make a previously aligned layout look wrong.

## References

- `references/scene-schema.md`: `scene.json` fields and coordinate rules
- `references/visio-component-map.md`: supported components and renderer intent
- `references/visio-export-flow.md`: Windows + Visio export path and current limitations
- `templates/style_profiles.json`: `paper_white` and `clean_white` rendering profiles
- `templates/examples/basic_flow.scene.json`: starter example
- `templates/examples/multimodal_paper_components.scene.json`: smoke example for multimodal spines, cuboids, trapezoids, and polygons
- `templates/examples/gan_loop_feedback.scene.json`: smoke example for smooth loop arrows, dashed regions, and dashed feedback paths
- `templates/examples/gan_tfr_full.scene.json`: canonical first-pass GAN/TFR template using `tfr_panel`, `loss_region`, `math_text`, smooth `loop_arrow`, and bundled backprop arrows
- `scripts/scene_complexity.py`: preflight report for large/dense figures before Visio rendering
- `scripts/font_inventory.py`: local Windows font inventory and preferred role fallback check
- `scripts/scene_autofix.py`: deterministic GAN/TFR local grammar upgrade pass before Visio rendering
- `docs/updates/2026-05-19-multimodal-paper-figure.md`: detailed analysis of a complex multimodal paper figure and the related component upgrade

## Current Boundaries

Version 1 is deliberately conservative:
- connectors support auto snap, orthogonal routing, and explicit points before full glue-aware connectors
- core flowchart shapes use local Visio masters when available, with controlled fallbacks
- export is handled by Visio after scene rendering rather than by translating raw SVG into Visio

That is the right tradeoff for a reusable first release.
