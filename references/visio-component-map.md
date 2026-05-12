# Visio Component Map

## Intent

This skill does not model all Visio shapes in version 1.

It uses a controlled semantic vocabulary so the analysis side stays stable while the renderer evolves from primitive geometry to richer stencil-aware rendering.

## Supported Node Types

| Type | Meaning | V1 renderer |
| --- | --- | --- |
| `process_box` | standard process/module block | rectangle |
| `rounded_process` | softer process block | rounded rectangle |
| `stacked_process` | repeated offset feature/map block shown as one semantic node | repeated editable rounded rectangles |
| `stacked_token` | compact token/vector stack shown as one semantic node | repeated editable small rectangles |
| `notched_block` | CNN/cutout module with visible notches | base shape plus editable notch cutouts |
| `feature_map_banded` | striped or columnar feature-map patch | editable bands and overlays |
| `feature_map_grid` | heatmap-like feature map with colored rows and shaded columns | editable cell rectangles plus separators |
| `merge_bus` | visible fan-in/fan-out bus or merge spine | editable line segments |
| `decision_diamond` | decision/judge node | rotated rectangle approximation |
| `terminator` | start/end block | high-rounding rectangle |
| `group_container` | grouping frame | dashed rectangle with optional title |
| `audit_region` | invisible logical review region for figures without visible frames | invisible rectangle used by validators/audit only |
| `text_pill` | small category/status label | pill |
| `ellipse_node` | oval/sum/add node | oval |
| `operator_node` | plus/multiply/tensor/add gate or explicit operator | editable oval with symbol text |
| `boundary_port` | module-frame input/output anchor | small visible/invisible port |
| `boundary_fanout` | parallel arrows emitted from a container boundary | editable line segments with arrowheads |
| `wave_signal` | waveform or signal snippet | editable polyline segments |
| `classifier_head` | AvgPool/Linear/output fan-out ending | editable composite blocks and lines |
| `text_block` | free text label, including rotated paper labels | invisible text box |
| `grid_matrix` | regular paper-style matrix/grid figure | editable cells plus grid lines |
| `bracket` | side grouping bracket or U-shaped paper marker | editable line segments |
| `junction_point` | tiny merge/fan anchor for many-to-one routing | tiny oval, usually invisible |
| `image_tile` | embedded secondary raster tile | imported image |
| `legend_block` | legend/annotation block | rectangle |

## Supported Edge Types

| Type | Meaning | V1 renderer |
| --- | --- | --- |
| `arrow_connector` | directional relation | straight line with arrow |
| `dynamic_connector` | routed relation | same renderer as arrow connector, with route support |
| `line_segment` | visual line/stub/bus segment without semantic arrow target | straight line without arrow |
| `join_connector` | source leg into a merge junction/bus, normally no arrowhead | routed line without arrow |
| `fork_connector` | branch out of a fan junction/bus, normally arrowed | routed line with arrow |
| `boundary_arrow` | arrow emitted by a group/frame boundary | forced-axis routed arrow |
| `residual_connector` | skip/residual loop relation | routed arrow, normally with explicit points |
| `residual_loop` | alias-style residual/skip loop relation | routed arrow, normally with explicit points |

## Why The Vocabulary Is Small

This is deliberate:
- localized Visio stencil names are messy
- early overfitting to master names will make the system fragile
- most flowchart reconstruction work only needs a limited component family

## Future Upgrade Path

The current renderer strategy is:

`scene type -> Visio master when useful, otherwise controlled primitive geometry`

Later we can extend it to:

`scene type -> locale-aware Visio master map -> drop master -> glue connectors`

That future change should not require a schema rewrite if node and edge types remain stable.

## Styling Guidance

Keep defaults restrained:
- white background
- dark neutral strokes
- limited accent colors
- moderate line weight
- clean typography

Use `paper_white` for paper figures and `clean_white` for polished product/process diagrams.

For exact replicas, prefer explicit `points` and `route` fields over manually nudging node boxes until arrows look right.

For merge/fan connectors, do not draw several arrows directly into the same box edge when the source has a visible shared trunk. Add a `junction_point` at the merge/fan position, route source arrows into that point without arrowheads, then route one final arrow to the destination.

For paper module figures, use `join_connector` for the no-arrow legs into the merge point and `fork_connector` for arrowed fan-out branches. Use `operator_node` when the source has a visible `+`, multiply, or tensor operator; do not replace that symbol with a generic ellipse or plain text.

`operator_node` is a dedicated circle-plus-symbol renderer. Put the operator in `symbol` or `text` on the same node; do not layer a separate `text_block` over an ellipse, because small baseline differences can push `×` or `⊗` outside the circle.

For arrows that enter or leave a dashed module frame, add `boundary_port` nodes on the frame edge. This prevents connector endpoints from drifting to the group box center and makes source-like horizontal stubs possible.

When an arrow should run horizontally from a frame boundary to a target component whose center is at a different height, use a side-ratio endpoint such as `feature_map:left@0.58`. Do not connect boundary ports to default component centers if the source shows a straight horizontal lane.

Use `boundary_arrow` when the visible source arrow starts at the dashed frame edge. Do not add an internal `line_segment` from the previous component to the frame unless that line is actually visible in the source. This prevents false long arrows that appear to originate from a vector strip or internal block.

For classifier outputs, distinguish two grammars:
- If arrows start after `Linear` from a shared internal trunk, use `classifier_head` with explicit `fanout_count`.
- If arrows start from the dashed classifier boundary, set `classifier_head.output_mode: "boundary"` or omit fanout fields, then use `boundary_fanout` on the frame edge.
- If `AvgPool` sits above `Linear`, set `classifier_head.orientation: "vertical"` so the internal arrow is drawn with a visible short shaft and a small arrowhead.

For short internal arrows, use `arrow_size: "small"` on normal edges and `internal_arrow_size: "tiny"` inside vertical classifiers. Visio's default arrowhead is too large for compact paper-module gaps and can look like a standalone triangle.

For waveform input strips and classifier endings, use `wave_signal`, `classifier_head`, and `boundary_fanout` before falling back to raster tiles. These components cover common paper-figure grammar while keeping the result editable.

For input modality brackets and paper-side grouping marks, use `bracket` instead of skinny rectangles. If the bracket has a middle merge arm, set `tick_positions: [0, 0.5, 1]`.

`group_container` should not be used as a connector endpoint. It represents a visual frame around a region only. If a figure needs a callout from a framed area, place a small `junction_point` on the desired border and connect to that.

`group_container` titles render as a small top label, not centered text. This keeps dashed module frames from covering internal arrows and blocks. Use `shape: "capsule"` or `shape: "rounded"` when the source has rounded dashed frames.

If the source has no visible module frames, use `audit_region` around logical areas instead of inventing dashed boxes. `audit_region` does not render, but it gives `scene_audit.py` a module boundary for child counts, incoming/outgoing edges, and topology review.

For cross-container flow, split the route:

`source node -> source boundary_anchor -> target boundary_anchor -> target node`

Set `allow_cross_container: true` only on the boundary-anchor bridge. This prevents arrows from visually leaking through dashed module frames.

For dense mini-flow diagrams, use `hv`/`vh` routes or explicit axis-aligned points. A connector must not pass through a non-endpoint node; route it around the node or move it into a bus lane.

For stacked feature maps or repeated blocks, use `stacked_process`. Do not model each offset layer as a separate node unless each layer has separate meaning; otherwise the validator will correctly treat connectors crossing hidden layers as topology defects.

For standalone stubs like a horizontal line from `Contact` toward `AvgPool` that should not visibly point into the `AvgPool` component, use `line_segment` with `from_point` and `to_point`. Use `arrow_connector` only when the source figure clearly has a directional arrowhead or terminates on a shape.

For module interiors such as AM-ResNet, add alignment metadata:
- `align_to_container: ["center_y"]` when a child should sit on the parent frame's midline.
- `align_group` plus `align_axis: "center_y"` when several internal components must share a row.

For exact replicas, set `metadata.fidelity: "exact"` and preserve the source aspect ratio before editing node coordinates. A scene that passes structural validation can still fail as a replica if it uses a tall canvas, random feature-map colors, generic CNN rectangles, or arrows that only satisfy graph semantics instead of the source's visible line grammar.

For exact replicas, prefer pixel-coordinate scenes. Set `page.units: "px"` with the source image dimensions and `target_width_in`; this keeps positions traceable to the source image and avoids drift caused by manual inch estimates.

For AM-ResNet features and similar paper heatmaps, use `feature_map_grid` rather than `feature_map_banded` overlays. Encode row colors and `column_shades` so dark vertical regions blend with the underlying row palette. Do not use opaque black overlays unless the source column is actually solid black.

Visio masters are useful but local. Use `scripts/enumerate_visio_masters.py` to inspect the installed master names on the current machine, then map only the few masters needed for the figure. Do not treat a master dump as a portable schema: stencil names differ by Office version, language, and installed templates.

Do not push chart junk or decorative gradient effects into the schema. The scene should encode structure first and style second.
