# 2026-05-19 Multimodal Paper Figure Update

## Source Figure Difficulty

The target figure is a high-difficulty paper-style multimodal framework diagram. It is not hard because of one complex shape; it is hard because many small visual grammars must stay consistent at the same time: modality lanes, availability masks, nested modules, 3D tensor blocks, trapezoid quality heads, formula annotations, and long cross-panel routes.

Overall difficulty: high.

Expected reconstruction mode: `metadata.fidelity: "exact"` or at least strict paper-figure mode with pixel-coordinate authoring.

## Module-Level Analysis

### 1. Left RGB / IR / SAR Inputs

Visual content:

- Three raster modality thumbnails.
- Large labels `RGB`, `IR`, `SAR`.
- Horizontal arrows from each image into availability flags.

Difficulty:

- Image content should not be redrawn as vector unless the user explicitly wants abstract placeholders.
- Use `image_tile` for the thumbnails.
- Each modality lane must remain horizontally aligned; center-to-center auto-routing can create diagonal drift.

Current skill capability:

- Already supported with `image_tile`, `text_block`, and explicit `arrow_connector` routes.

Recommended encoding:

- Use one `image_tile` per input.
- Use `text_block` for modality labels.
- Use `route: "horizontal"` and side endpoints such as `rgb_img:right@0.5`.

### 2. Availability Flags and Alpha Labels

Visual content:

- Three gray `1 or 0` boxes.
- Small math labels `alpha_RGB`, `alpha_IR`, `alpha_SAR`.
- These feed into repeated modality ports.

Difficulty:

- Tiny math labels are easy to shift or overlap.
- The boxes look simple, but their lane alignment is important.

Current skill capability:

- Already supported with `process_box`, `text_pill`, and `text_block`.

Recommended encoding:

- Use `process_box` or `text_pill` for `1 or 0`.
- Use separate `text_block` for each alpha label.
- Treat the three lanes as an `audit_region` so `scene_audit.py` checks missing or wrong lane arrows.

### 3. Vertical Availability Mask / Shared Spine

Visual content:

- A tall gray vertical bar.
- Three green ports `P_RGB`, `P_IR`, `P_SAR` placed along the spine.
- A vertical math label `a_m`.

Difficulty:

- Before this update, this required a plain rectangle plus three separately positioned boxes.
- Separate boxes make topology and alignment fragile.
- Port labels need to be visually attached to the spine but still act as lane targets.

Skill update:

- Added `modality_spine`.

Recommended encoding:

- Use one `modality_spine` node with `ports`.
- Use external anchors or side endpoints when a specific port needs independent routing.
- Use `text_block` with `angle_deg` for the vertical `a_m` label.

### 4. Modality Projection Block

Visual content:

- A large blue rounded container.
- Nested shallow shared backbone.
- Stacked 3D-like blocks: `DWConv`, `1x1 Conv`, `BN`, `SiLU`, `Residual Unit`.
- A vertical formula block `{F_m^s | m in G}`.

Difficulty:

- Nested containers can cause arrows to connect to frames instead of components.
- The stacked backbone blocks need a consistent 3D offset.
- The formula block is a label, not a semantic connector target.

Current skill capability:

- Mostly supported with `group_container`, `stacked_process`, `text_block`, and explicit connectors.

Recommended encoding:

- Use `group_container` for the blue module and the inner backbone.
- Use `stacked_process` for each 3D-looking internal block.
- Use `text_block` for the formula brace.
- Use `audit_region` around the projection module to catch missing internal arrows.

### 5. Modality-Related Impact Factor

Visual content:

- A large green 3D cuboid labeled `c_m`.
- It receives the availability mask output and emits feature-like signals.

Difficulty:

- A flat rectangle loses the 3D tensor cue.
- The right side face and top face must remain aligned with the front face.

Skill update:

- Added `cuboid_node`.

Recommended encoding:

- Use `cuboid_node` with front, top, and side colors.
- Keep connectors attached to the front semantic bounds.

### 6. Modality Quality Description Panel

Visual content:

- A titled container.
- Text block `Pooling Statistics [GAP,GMP, Gradient,Std]`.
- A gray right-facing quality head.
- A formula vector `q = [q_RGB, q_IR, q_SAR]`.

Difficulty:

- The quality head is not a rectangle; it is a directional trapezoid/wedge.
- The formula vector is visually important but not a flow node.
- Long arrows enter from the left and exit to a vector annotation.

Skill update:

- Added `trapezoid_node`.

Recommended encoding:

- Use `text_block` for pooling statistics and formula vectors.
- Use `trapezoid_node` for the quality head.
- Do not connect arrows to the formula text unless the source shows it as a block.

### 7. Environmental Perception Modeling Panel

Visual content:

- Small blue feature cuboids.
- An orange environment response extractor wedge.
- GAP and GMP grid icons.
- An orange aggregation quality-aware wedge.
- Multiple horizontal arrows.

Difficulty:

- Several small blocks share nearly the same horizontal lanes.
- The two orange wedge modules face opposite directions visually.
- GAP/GMP grids must look like structured feature maps, not random colored squares.

Skill update:

- `cuboid_node` covers small depth blocks.
- `trapezoid_node` covers extractor and aggregation wedges.
- Existing `feature_map_grid` covers GAP/GMP icons.

Recommended encoding:

- Use `cuboid_node` for feature tensors.
- Use `trapezoid_node` with `orientation: "right"` or `orientation: "left"` for wedges.
- Use `feature_map_grid` for GAP/GMP.
- Use explicit `points` for cross-lane arrows.

### 8. Structural State Encoding / Conditional Variable Generation Panel

Visual content:

- A large panel with Cross-modal Structure Extractor, GAP feature map, Conditional Variable Generator, Environment Encoder, and long routed arrows.
- The environment encoder looks like a symmetric hourglass made of two opposing wedges.

Difficulty:

- This panel has the highest routing risk.
- Long black routes run around components and cross large empty regions.
- The hourglass encoder is an unusual custom polygon.
- Several arrow labels are tiny and easy to attach to the wrong route.

Skill update:

- Added `polygon_node` as a fallback for hourglass and asymmetric paper shapes.
- Existing `boundary_port`, `junction_point`, `line_segment`, and `arrow_connector` should be used for long routes.

Recommended encoding:

- Build the hourglass from two `polygon_node` nodes plus a thin center rectangle if needed.
- Use `junction_point` for route bends.
- Use `line_segment` for visual stubs with no arrowhead.
- Use `route: "hv"`, `route: "vh"`, or explicit `points` for every long route.

## Can The Current Skill Make This Figure?

Before this update:

- The overall diagram could be approximated.
- The major containers, input images, standard boxes, feature maps, formula labels, and arrows were already possible.
- The weak points were 3D cuboids, trapezoid heads, vertical modality spines with ports, and unusual hourglass/polygon encoders.

After this update:

- The figure is within the intended scope of Visiomaster.
- It still requires careful scene authoring and visual review.
- It should be treated as high difficulty, not a one-pass automatic conversion.

## Implemented Skill Changes

Added node types:

- `polygon_node`: generic editable polygon fallback for unusual paper geometry.
- `trapezoid_node`: directional wedge/trapezoid modules such as quality heads and extractors.
- `cuboid_node`: editable 3D tensor/impact-factor block with front, top, and side faces.
- `modality_spine`: vertical shared-response or availability-mask spine with repeated modality ports.

Updated files:

- `scripts/scene_to_visio.py`: renderer support for the new node types.
- `scripts/scene_validate.py`: validation for polygon points, trapezoid orientation/taper, and modality spine ports.
- `templates/visio_components.json`: component vocabulary additions.
- `templates/style_profiles.json`: default paper/clean styles for new components.
- `references/scene-schema.md`: schema examples and usage rules.
- `references/visio-component-map.md`: component mapping and modeling guidance.
- `SKILL.md`: execution rules for multimodal paper figures.
- `templates/examples/multimodal_paper_components.scene.json`: small smoke scene covering the new components.

## Authoring Checklist For Similar Figures

- Use pixel coordinates and preserve source aspect ratio.
- Create visible `group_container` nodes for titled panels.
- Add `audit_region` nodes for logical lanes inside complex panels.
- Use `image_tile` for real RGB/IR/SAR thumbnails.
- Use `modality_spine` for shared modality bars.
- Use `cuboid_node` for 3D tensor blocks.
- Use `trapezoid_node` for quality/extractor/aggregation wedges.
- Use `polygon_node` only for unusual geometry such as hourglass encoders.
- Keep formula labels as `text_block`.
- Route long arrows with explicit `points`; avoid auto center-to-center routes in this figure.
