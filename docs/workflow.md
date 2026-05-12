# Visiomaster Workflow

This document describes the reconstruction loop used by Visiomaster.

## Overall Flow

```mermaid
flowchart TD
  A["Input<br/>diagram image / paper figure / product flow"] --> B["Visual analysis<br/>layout, modules, labels, styles"]
  B --> C["Scene authoring<br/>scene.json"]
  C --> D["Validation<br/>scene_validate.py"]
  D --> E{"Schema, ids,<br/>routes, fidelity OK?"}
  E -- "No" --> C
  E -- "Yes" --> F["Module audit<br/>scene_audit.py"]
  F --> G{"Topology and<br/>module checklist OK?"}
  G -- "No" --> C
  G -- "Yes" --> H["Render<br/>scene_to_visio.py"]
  H --> I["Visio document<br/>.vsdx"]
  H --> J["Exports<br/>.svg / .png"]
  J --> K["Visual QA<br/>compare PNG with source"]
  K --> L{"Local details<br/>match source?"}
  L -- "No" --> C
  L -- "Yes" --> M["Final delivery<br/>editable Visio + exports"]
```

## Scene Authoring Loop

```mermaid
flowchart LR
  A["Source image"] --> B["Identify visible modules"]
  A --> C["Identify logical modules"]
  B --> D["group_container"]
  C --> E["audit_region"]
  D --> F["Place child nodes"]
  E --> F
  F --> G["Add explicit ports and junctions"]
  G --> H["Add edges with route/points"]
  H --> I["Validate and audit"]
  I --> J{"Issues?"}
  J -- "Yes" --> F
  J -- "No" --> K["Render"]
```

Use `group_container` when the source has a visible module boundary. Use `audit_region` when the source has no visible boundary but the figure still needs local review, such as a residual block, classifier head, attention module, or feature extraction lane.

## Quality Gates

```mermaid
flowchart TD
  A["Validation gate"] --> A1["JSON loads"]
  A --> A2["Node and edge ids resolve"]
  A --> A3["Supported component types"]
  A --> A4["Route-quality warnings"]
  A --> A5["Exact-mode aspect ratio"]

  B["Audit gate"] --> B1["Module child count"]
  B --> B2["Incoming/outgoing edges"]
  B --> B3["Internal topology"]
  B --> B4["Boundary ports and fanout"]
  B --> B5["Cross-container bridges"]

  C["Visual QA gate"] --> C1["Canvas ratio"]
  C --> C2["Container bounds"]
  C --> C3["Arrow grammar"]
  C --> C4["Special shapes/operators"]
  C --> C5["Feature map coloring"]
```

## Practical Rule

Do not judge complex reconstructions only by whole-image similarity. Review each module independently, because the most common failures are local: slightly shifted nodes, diagonal arrows where the source is horizontal, a connector glued to the wrong component, or a boundary output drawn from an internal block.
