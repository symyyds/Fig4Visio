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

If the layout is close to a standard process flow, you can seed from the built-in example:

```powershell
python ${SKILL_DIR}\scripts\image_to_scene.py --image <source.png> --template basic-flow --output <scene.json>
```

### 3. Validate scene data

Before touching Visio, validate structure and references:

```powershell
python ${SKILL_DIR}\scripts\scene_validate.py <scene.json>
```

If validation fails, fix the scene first. Do not guess around broken ids or unsupported types inside the renderer.

For complex paper figures with many modules, also generate a module-level audit report:

```powershell
python ${SKILL_DIR}\scripts\scene_audit.py <scene.json>
```

Use the audit report to review every `group_container` as a separate region: child count, labels, colors, internal arrows, incoming arrows, outgoing arrows, and whether cross-module arrows start from a boundary or from an internal component. Treat unchecked audit items as real defects before final export.

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
- `audit_region`
- `text_pill`
- `ellipse_node`
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

## References

- `references/scene-schema.md`: `scene.json` fields and coordinate rules
- `references/visio-component-map.md`: supported components and renderer intent
- `references/visio-export-flow.md`: Windows + Visio export path and current limitations
- `templates/style_profiles.json`: `paper_white` and `clean_white` rendering profiles
- `templates/examples/basic_flow.scene.json`: starter example

## Current Boundaries

Version 1 is deliberately conservative:
- connectors support auto snap, orthogonal routing, and explicit points before full glue-aware connectors
- core flowchart shapes use local Visio masters when available, with controlled fallbacks
- export is handled by Visio after scene rendering rather than by translating raw SVG into Visio

That is the right tradeoff for a reusable first release.
