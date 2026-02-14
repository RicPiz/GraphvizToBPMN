# Graphviz to BPMN 2.0 Converter

This repository contains a Python converter that transforms process models from Graphviz DOT (`.gv`) into BPMN 2.0 XML (`.bpmn`) with diagram coordinates.

## Why this project exists

It is not unusual to see process variants as DOT graphs, while most BPM tooling consumes BPMN 2.0 XML (for example bpmn.io-based tools and BPM suites). The converter bridges that gap by:

- parsing DOT process structure,
- reconstructing BPMN semantics (events, tasks, gateways, sequence flows), and
- generating BPMN Diagram Interchange (DI) coordinates so diagrams are immediately viewable.

The tests were performed on the [MaD dataset](https://drive.google.com/drive/u/0/folders/1n0K9BmiDsXYCqB796MVebYBgWX2ruZpW), which contains 15 business categories and a total of 30,000 BPM-description pairs. The publication related to this work is [MaD: A Dataset for Interview-based BPM in Business Process Management](https://ieeexplore.ieee.org/document/10191898/figures#figures).
Here are the results for one of the business categories.

## What is in this repository

- `gv_to_bpmn.py`: converter implementation.
- One folder containing `.gv` and `.txt` files from MaD.
- Generated `.bpmn` examples for conversion output.

## Input conventions expected by the converter

The parser infers BPMN types from node names/shapes used in the dataset.

| Graphviz pattern | BPMN output |
| --- | --- |
| `START_NODE` | `startEvent` |
| `END_NODE` | `endEvent` |
| `shape=box` | `task` |
| `AND_SPLIT--...` / `AND_JOIN--...` | `parallelGateway` |
| `OR_SPLIT--...` / `OR_JOIN--...` | `inclusiveGateway` |
| `XOR_SPLIT--...` / `XOR_JOIN--...` | `exclusiveGateway` |

Gateway labels are cleaned from encoded names by removing:

- trailing random 4-char suffixes (for example `_R91M`), and
- leading numeric prefixes (for example `3. `).

## Algorithm

The implementation follows a 4-stage pipeline, derived from the pseudo-code description in the paper [Size matters less: how fine-tuned small LLMs excel in BPMN generation](https://link.springer.com/article/10.1186/s43067-025-00288-9).

### 1. Parse DOT into a directed graph

- Uses `pydot.graph_from_dot_data`.
- Recursively traverses subgraphs/clusters so nested nodes and edges are not lost.
- Creates a `networkx.DiGraph` with node attributes (`type`, `label`) and edge labels.

### 2. Compute layout

- Finds a start node (`START_NODE` if present).
- Assigns x-levels using longest-path style propagation over:
  - topological order for DAGs,
  - BFS fallback when cycles exist.
- Forces `END_NODE` to the rightmost column (`max_level + 1`).
- Detects backward edges (potential loops) where `source_level >= target_level`.
- Assigns y-positions with branch spreading at splits and same-line continuation for single successors.
- Aligns gateways:
  - join gateways centered on predecessor y-values,
  - split gateways centered on successor y-values.
- Resolves overlaps iteratively (multi-pass per level, max 10 iterations).
- Produces `(x, y, width, height)` per node with fixed dimensions per BPMN element type.

### 3. Generate BPMN process + DI

- Creates BPMN 2.0 `definitions` and `process` XML.
- Maps graph nodes to BPMN elements.
- Creates `sequenceFlow` edges and carries edge labels to `name`.
- For labeled edges leaving an XOR split, also writes a BPMN `conditionExpression`.
- Writes BPMN DI shapes (`BPMNShape`) from layout coordinates.
- Writes BPMN DI edges (`BPMNEdge`) with waypoints:
  - forward edges: straight right-to-left connection between source/target shapes,
  - backward edges: routed with extra waypoints below the flow to visually separate loops.

### 4. Batch conversion workflow

- Single-file mode: convert one `.gv`.
- Directory mode: recursively find all `.gv` files and generate sibling `.bpmn` files.

## Implementation difference vs. the reference pseudo-code

Relative to the pseudo-code version of the algorithm, loop handling is intentionally implemented differently:

- Reference idea: adjust node positions when backward edges exist.
- This implementation: keep node layout stable and route backward edges with additional waypoints in DI.

Why this choice:

- preserves clean left-to-right placement,
- makes loops visually explicit,
- avoids perturbing already-computed node alignment.

## Installation

```bash
python3 -m pip install pydot networkx
```

## Usage

Convert one file:

```bash
python3 gv_to_bpmn.py --single account_payable_process/account_payable_process_0.gv
```

Convert every `.gv` under a directory recursively:

```bash
python3 gv_to_bpmn.py .
```

Verbose mode:

```bash
python3 gv_to_bpmn.py . --verbose
```

Output behavior:

- Each input `path/to/model.gv` produces `path/to/model.bpmn`.
- Existing output files are overwritten.

## Limitations

This converter currently targets the dataset conventions above. It does not model advanced BPMN constructs such as pools/lanes, subprocesses, message flows, timers, or event-based gateways.