# Scene Schema

## Purpose

`scene.json` is the intermediate semantic format between image analysis and Visio rendering.

The format is intentionally small. It should describe structure, layout, style, and references without exposing Visio COM details directly.

## Coordinate Rules

- Units: inches by default
- Page origin: top-left
- Node `x` and `y`: top-left corner
- Node `w` and `h`: width and height
- Renderer is responsible for converting to Visio's bottom-left coordinate system

For exact replicas, author in source pixels instead of eyeballed inches:

```json
{
  "page": {
    "width": 1514,
    "height": 538,
    "units": "px",
    "origin": "top-left",
    "target_width_in": 13.333,
    "background": "#FFFFFF"
  }
}
```

The renderer scales all node coordinates, `from_point`, `to_point`, and edge `points` to inches. `target_height_in` is optional; if omitted it is calculated from the source aspect ratio.

## Top-Level Structure

```json
{
  "version": "0.1",
  "metadata": {},
  "page": {},
  "nodes": [],
  "edges": [],
  "assets": []
}
```

## `page`

Required fields:

```json
{
  "width": 13.333,
  "height": 7.5,
  "units": "in",
  "origin": "top-left",
  "background": "#FFFFFF"
}
```

## `metadata`

Recommended fields:

```json
{
  "title": "Radar Sorter Overview",
  "created_by": "visiomaster.image_to_scene",
  "style_profile": "paper_white",
  "fidelity": "exact",
  "source_image": "C:/path/source.png",
  "source_aspect_ratio": 2.817,
  "style_reference": "C:/path/style.jpg",
  "notes": [
    "Main structure should stay editable.",
    "Image tiles may remain raster when they are secondary."
  ]
}
```

For 1:1 replica tasks, set `fidelity` to `exact`, and provide either `source_image` or `source_aspect_ratio`. The validator compares the page aspect ratio against the source so a wide paper figure is not accidentally rebuilt on a taller canvas.

## `nodes`

Each node must contain:

```json
{
  "id": "node-id",
  "type": "process_box",
  "x": 1.0,
  "y": 1.2,
  "w": 2.4,
  "h": 0.9,
  "text": "Label"
}
```

Optional node fields:

```json
{
  "style": {
    "fill": "#FFFFFF",
    "line": "#111827",
    "line_weight_pt": 1.25,
    "text_color": "#111827",
    "font_family": "Times New Roman",
    "font_weight": "regular",
    "font_size_pt": 16,
    "line_dash": "solid",
    "rounding_in": 0.12,
    "angle_deg": 90,
    "text_angle_deg": 0
  },
  "asset_ref": "asset-id",
  "container_id": "optional-parent-container-id",
  "align_to_container": ["center_y"],
  "align_group": "row-1",
  "align_axis": "center_y",
  "align_tolerance_in": 0.05,
  "z": 10
}
```

Use alignment fields when reconstructing module boxes such as AM-ResNet. If a component should sit on the visual midline of its parent block, set `container_id` and `align_to_container: ["center_y"]`. If several sibling nodes should share one row, give them the same `align_group` and `align_axis`.

Supported node `type` values are defined in `templates/visio_components.json`.

### `stacked_process`

Use `stacked_process` for repeated offset feature maps or stacked module blocks that behave as one semantic node.

```json
{
  "id": "rgb_backbone",
  "type": "stacked_process",
  "x": 3.8,
  "y": 2.2,
  "w": 0.64,
  "h": 0.76,
  "text": "Backbone",
  "layers": 4,
  "stack_dx_in": -0.045,
  "stack_dy_in": 0.035,
  "style": {
    "fill": "#8FB2E6",
    "line": "#3A6DA8",
    "rounding_in": 0.05
  }
}
```

Do not author each visible offset layer as an independent node unless each layer is separately connected or labeled. Use one `stacked_process` so connector validation sees one semantic target.

Use `stacked_token` for smaller token/vector stacks. It uses the same fields as `stacked_process`, but has tighter default spacing and smaller type.

### `notched_block`

Use `notched_block` for CNN-like modules with visible cutouts or teeth. It renders an editable base rectangle plus editable white notch shapes.

```json
{
  "id": "cnn",
  "type": "notched_block",
  "x": 320,
  "y": 290,
  "w": 120,
  "h": 190,
  "text": "CNN",
  "notches": [
    {"x": 0.52, "y": 0.30, "w": 0.36, "h": 0.16, "shape": "diamond"},
    {"x": 0.52, "y": 0.74, "w": 0.36, "h": 0.16, "shape": "diamond"}
  ]
}
```

### `feature_map_banded`

Use `feature_map_banded` for paper feature maps with horizontal stripes, vertical dark columns, or fixed color bands. Do not use random grids when the source has recognizable bands.

```json
{
  "id": "wav_features",
  "type": "feature_map_banded",
  "x": 650,
  "y": 360,
  "w": 150,
  "h": 60,
  "bands": [
    {"fill": "#B7DCEB", "size": 1},
    {"fill": "#F8E49B", "size": 1},
    {"fill": "#B7DCEB", "size": 1},
    {"fill": "#C9C0D8", "size": 1}
  ],
  "overlays": [
    {"x": 0.52, "y": 0, "w": 0.18, "h": 1, "fill": "#1E1E1E"}
  ]
}
```

### `feature_map_grid`

Use `feature_map_grid` for AM-ResNet features, heatmap-like feature maps, and small paper feature blocks where each vertical column preserves the row colors but has different brightness or transparency. This avoids the common failure where semi-transparent dark columns become opaque black bars.

```json
{
  "id": "am_resnet_features",
  "type": "feature_map_grid",
  "x": 5.80,
  "y": 1.80,
  "w": 1.55,
  "h": 0.86,
  "rows": 6,
  "cols": 9,
  "row_colors": [
    "#F2A66F",
    "#A8D7E5",
    "#C8D9C2",
    "#F3E889",
    "#9BC6D9",
    "#F2A66F"
  ],
  "column_shades": [0.0, 0.20, 0.0, 0.45, 0.70, 0.45, 0.0, 0.20, 0.0],
  "max_shade": 0.58,
  "show_column_lines": true,
  "show_row_lines": false
}
```

`column_shades` values are normalized from `0` to `1`; the renderer blends the row color toward `shade_color` instead of laying an opaque black rectangle over the feature map. Use `column_weights` if the source has uneven column widths.

### `merge_bus`

Use `merge_bus` for visible bus/spine merges or fan-in/fan-out trunks. It is a visible topology component, unlike invisible `junction_point`.

```json
{
  "id": "concat_bus",
  "type": "merge_bus",
  "x": 980,
  "y": 250,
  "w": 40,
  "h": 80,
  "orientation": "vertical",
  "side": "left",
  "port_positions": [0, 1],
  "port_length_in": 0.16
}
```

`group_container` labels are rendered as top labels by default. Optional title controls:

```json
{
  "shape": "capsule",
  "corner_radius_in": 0.38,
  "max_rounding_in": 0.45,
  "title_align": 0,
  "title_pad_x_in": 0.08,
  "title_pad_y_in": 0.02,
  "title_font_size_pt": 13
}
```

Use `shape: "capsule"` or `shape: "rounded"` when the source has paper-style rounded dashed frames. Keep plain rectangles only when the source frame is visibly square.

### `audit_region`

Use `audit_region` for figures that have no visible dashed modules but still need module-level review. It behaves like an invisible container: it does not render a frame or label, but `scene_validate.py` and `scene_audit.py` use it to group child nodes and edges.

```json
{
  "id": "residual_block_1",
  "type": "audit_region",
  "x": 0.80,
  "y": 1.10,
  "w": 3.20,
  "h": 1.70,
  "label": "Residual block 1"
}
```

Use this when the source visually contains a logical block such as residual network, attention, classifier, feature extraction chain, or repeated BN/R/M/C sequence, but no visible boundary is drawn. Do not connect edges to `audit_region`; use normal component endpoints or `boundary_port` only when a visible boundary output exists.

### `operator_node`

Use `operator_node` for explicit paper-figure operators such as plus, multiply, tensor product, add, concat markers, or attention gates. Do not fake these as text floating over a connector.

```json
{
  "id": "residual_add",
  "type": "operator_node",
  "x": 6.10,
  "y": 2.42,
  "w": 0.22,
  "h": 0.22,
  "symbol": "⊗"
}
```

`operator_node` renders the circle and the symbol as a controlled pair, with the symbol centered in the circle. Prefer `symbol` over a separate `text_block`. Keep `w` and `h` equal for exact replicas. Optional tuning fields: `symbol_font_size_pt`, `symbol_font_family`, `symbol_offset_x_in`, `symbol_offset_y_in`, and `symbol_inset_in`.

### `boundary_port`

Use `boundary_port` for module-frame entry/exit anchors. This is the preferred fix when a connector should touch the dashed frame boundary without connecting to the `group_container` itself.

```json
{
  "id": "am_resnet_out",
  "type": "boundary_port",
  "container_id": "am_resnet",
  "side": "right",
  "shape": "none",
  "x": 7.52,
  "y": 2.95,
  "w": 0.04,
  "h": 0.04
}
```

Set `shape` to `circle`, `square`, `tick`, or `none`. Use `visible: false` or `shape: "none"` for an invisible but editable routing anchor.

### `wave_signal`

Use `wave_signal` for waveform inputs or signal snippets. It renders as editable line segments inside the node box.

```json
{
  "id": "audio_wave",
  "type": "wave_signal",
  "x": 0.55,
  "y": 2.20,
  "w": 1.20,
  "h": 0.42,
  "cycles": 3,
  "point_count": 64
}
```

For source-faithful waveform shapes, provide normalized `samples` in `[-1, 1]`.

### `classifier_head`

Use `classifier_head` for common paper endings such as `AvgPool -> Linear`. It is a compact editable composite; for 1:1 replicas with unusual spacing, you can still break it into separate `process_box`, `junction_point`, and connector nodes.

Do not let `classifier_head` draw internal fan-out when the source shows arrows starting on the dashed classifier frame. In that case set `output_mode: "boundary"` or omit `fanout_count`, then add a `boundary_fanout` node on the frame's right side.

```json
{
  "id": "cls_head",
  "type": "classifier_head",
  "x": 10.45,
  "y": 2.15,
  "w": 1.85,
  "h": 0.95,
  "labels": ["AvgPool", "Linear"],
  "orientation": "vertical",
  "vertical_block_gap_in": 0.18,
  "internal_arrow_size": "tiny",
  "output_mode": "boundary"
}
```

Use `fanout_count` and `output_labels` only when the source truly has a shared internal branch after the last classifier block.

For vertical classifier blocks, use `orientation: "vertical"` so the renderer creates a visible short shaft between `AvgPool` and `Linear`. Do not model this with an extremely short generic edge; Visio arrowheads can consume the whole line segment if the arrow size is not reduced. Use `internal_arrow_size: "tiny"` for compact paper figures.

### `boundary_fanout`

Use `boundary_fanout` when several arrows originate from a dashed container boundary, as in many paper classifier outputs. It draws parallel editable arrows from the frame edge outward, instead of inventing a central merge point after `Linear`.

```json
{
  "id": "classifier_outputs",
  "type": "boundary_fanout",
  "container_id": "classifier",
  "side": "right",
  "x": 12.28,
  "y": 2.05,
  "w": 0.62,
  "h": 1.70,
  "branch_positions": [0.08, 0.38, 0.68, 0.92],
  "labels": ["N", "S", "V", "F"]
}
```

For right-side output arrows, set `x` to the container's right boundary and `w` to the outward arrow length. `branch_positions` values in `[0, 1]` are normalized over `h`; larger values are treated as absolute offsets in scene units.

### `grid_matrix`

Use `grid_matrix` for convolution kernels, receptive fields, checkerboard masks, and paper figures with regular cells.

```json
{
  "id": "kernel_a",
  "type": "grid_matrix",
  "x": 1.0,
  "y": 0.8,
  "w": 2.3,
  "h": 2.3,
  "rows": 9,
  "cols": 9,
  "index_base": 0,
  "colored_cells": [
    {"row": 2, "col": 2, "fill": "#2F7F91"},
    {"row": 4, "col": 4, "fill": "#F07A00"}
  ],
  "style": {
    "cell_fill": "#FFFFFF",
    "grid_line": "#000000",
    "grid_line_weight_pt": 1.0
  }
}
```

`colored_cells` also accepts compact entries: `[row, col, fill]`.

The renderer creates editable cell rectangles and separate grid lines inside Visio. Do not hand-place 81 small boxes when one `grid_matrix` can express the structure.

### `bracket`

Use `bracket` for modality grouping marks, braces, and paper-style side brackets. Do not fake these with ultra-thin `process_box` rectangles.

```json
{
  "id": "input_modalities_bracket",
  "type": "bracket",
  "x": 1.65,
  "y": 2.4,
  "w": 0.35,
  "h": 3.05,
  "orientation": "right",
  "tick_positions": [0, 0.5, 1],
  "style": {
    "line": "#333333",
    "line_weight_pt": 1.1
  }
}
```

`orientation` values:
- `right`: spine on the right edge, arms extend left; visually like `]`
- `left`: spine on the left edge, arms extend right; visually like `[`
- `down`: bottom spine, arms extend upward
- `up`: top spine, arms extend downward

Use `tick_positions` when the source bracket has a middle merge arm. Values are normalized from `0` to `1` along the bracket span. A left-side modality merge often needs `[0, 0.5, 1]`, not just `[0, 1]`.

### `junction_point`

Use `junction_point` for explicit 2-to-1, many-to-one, and fan-out routing. It is usually invisible and tiny, but gives connectors a semantic merge/fan point.

```json
{
  "id": "evidence_merge",
  "type": "junction_point",
  "x": 8.86,
  "y": 7.74,
  "w": 0.04,
  "h": 0.04,
  "role": "merge",
  "style": {
    "fill": "none",
    "line": "none"
  }
}
```

Pattern:

```json
[
  {"id": "rgb_to_merge", "type": "arrow_connector", "from": "z_rgb:right", "to": "evidence_merge:center", "style": {"end_arrow": "none"}},
  {"id": "ir_to_merge", "type": "arrow_connector", "from": "z_ir:right", "to": "evidence_merge:center", "style": {"end_arrow": "none"}},
  {"id": "merge_to_layer", "type": "arrow_connector", "from": "evidence_merge:center", "to": "evidence_layer:left"}
]
```

Do not connect arrows directly to `group_container`; containers are frames, not flow targets.

For cross-container connectors, place boundary anchors on the relevant frame edges:

```json
{
  "id": "feature_out_portal",
  "type": "junction_point",
  "role": "boundary_anchor",
  "container_id": "feature_container",
  "x": 7.60,
  "y": 2.68,
  "w": 0.03,
  "h": 0.03
}
```

Then split the connector into internal, bridge, and internal segments. Mark only the bridge segment with `allow_cross_container: true`.

## `edges`

Each edge must contain:

```json
{
  "id": "edge-id",
  "type": "arrow_connector",
  "from": "node-a",
  "to": "node-b"
}
```

For a purely visual line segment that should not snap to a component, use point endpoints:

```json
{
  "id": "contact_to_avgpool_stub",
  "type": "line_segment",
  "from_point": [4.10, 2.35],
  "to_point": [5.20, 2.35],
  "route": "straight",
  "style": {
    "end_arrow": "none"
  }
}
```

This is the correct representation when the source figure shows a horizontal line only, not an arrow pointing into `AvgPool`.

Optional edge fields:

```json
{
  "label": "Optional label",
  "allow_diagonal": false,
  "allow_cross_container": false,
  "style": {
    "line": "#64748B",
    "line_weight_pt": 1.25,
    "line_dash": "solid",
    "end_arrow": "triangle"
  },
  "z": 100
}
```

Endpoint syntax currently supports:
- `node-id`
- `node-id:left`
- `node-id:right`
- `node-id:top`
- `node-id:bottom`
- `node-id:center`
- `node-id:left@0.62`
- `node-id:right@0.58`

If no side is given, the renderer auto-selects a side based on relative position.

Use `@ratio` side anchors when a line must hit a component edge at the same visual height as a boundary port or bus lane. For `left`/`right`, the ratio is vertical from top `0` to bottom `1`; for `top`/`bottom`, it is horizontal from left `0` to right `1`. This is the preferred way to keep a frame-to-feature-map arrow horizontal without moving the target component.

Point endpoint fields:
- `from_point`: `[x, y]`
- `to_point`: `[x, y]`

Point endpoints may be combined with node endpoints, for example from `contact:right` to a free `to_point`, or from a free `from_point` to `avgpool:left`. Use this only when the figure's geometry needs a free-floating segment or bus stub.

Routing fields:

```json
{
  "route": "auto",
  "points": [[1.2, 3.5], [7.4, 3.5]],
  "snap_tolerance_in": 0.2
}
```

`route` values:
- `auto`: snap nearly aligned endpoints and use right-angle routes for opposite sides
- `straight`: draw one direct segment
- `horizontal`: force a horizontal line from source x/y to target x at the source y
- `vertical`: force a vertical line from source x/y to target y at the source x
- `orthogonal`, `elbow`, `right_angle`: force right-angle routing
- `hv`, `horizontal_then_vertical`: horizontal segment first, then vertical
- `vh`, `vertical_then_horizontal`: vertical segment first, then horizontal

Use `join_connector` for source-to-merge legs that should reach a shared junction or bus without arrowheads. Use `fork_connector` for fan-out branches leaving a junction or bus. These edge types keep fan-in/fan-out topology explicit instead of drawing several unrelated arrows into one box edge.

Use `residual_connector` or `residual_loop` for skip/residual loops. They render like arrow connectors but signal that the route must preserve loop topology and should normally use explicit axis-aligned `points`.

Use `boundary_arrow` when the source arrow starts from a group/frame boundary rather than from the last internal component:

```json
{
  "id": "frame_to_features",
  "type": "boundary_arrow",
  "from": "module_out:center",
  "to": "features:left@0.58",
  "route": "horizontal",
  "allow_cross_container": true
}
```

Do not add an internal line from `vector:right` to `module_out:center` unless that internal line is visible in the source. The boundary arrow should usually be the only visible external output.

Use `points` for exact paper-style residual paths, skip connections, and hand-tuned replicas.

Arrow sizing:

```json
{
  "style": {
    "arrow_size": "small"
  }
}
```

`arrow_size` accepts `tiny`, `small`, `medium`, `large`, or an integer code. The renderer also shrinks arrowheads on very short segments so small internal arrows remain a line plus a head instead of a head-only mark.

Routing quality rules:
- Do not use diagonal `straight` lines for flow connectors unless the source is a real callout; set `allow_diagonal: true` only for intentional callouts.
- Do not force every line to terminate on a shape. If the source has a standalone horizontal or vertical stub, use `line_segment` with `from_point` and `to_point`.
- If an `orthogonal` edge has `points`, each adjacent point pair must share either `x` or `y`; otherwise the renderer will still draw a diagonal segment.
- Keep intra-module connectors inside their `group_container`.
- For connectors between modules, use `junction_point` with `role: boundary_anchor` at the frame edges and split the route. Do not let one long edge run through multiple dashed frames.
- If a route crosses a non-endpoint process node, move it to a bus lane or add explicit points around the node.

## `assets`

Assets are optional and mainly used by `image_tile` nodes.

```json
{
  "id": "asset-map",
  "kind": "image",
  "path": "C:/path/map.png"
}
```

## Example

See `templates/examples/basic_flow.scene.json` for a working starter file.
