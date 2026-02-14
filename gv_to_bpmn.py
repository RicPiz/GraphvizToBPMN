#!/usr/bin/env python3
"""
GV to BPMN 2.0 Converter

Converts Graphviz DOT (.gv) files to BPMN 2.0 XML (.bpmn) files with full visual layout.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

try:
    import pydot
except ImportError:
    print("Error: pydot is required. Install with: pip install pydot")
    sys.exit(1)

try:
    import networkx as nx
except ImportError:
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)


# Layout constants
X_SPACING = 180  # Horizontal spacing between levels
Y_SPACING = 100  # Vertical spacing between nodes at same level
START_X = 50     # Starting X coordinate
START_Y = 100    # Starting Y coordinate

# Element dimensions
TASK_WIDTH = 100
TASK_HEIGHT = 80
GATEWAY_SIZE = 50
EVENT_SIZE = 36


def parse_gv_to_graph(gv_content: str) -> nx.DiGraph:
    """
    Parse GV content and build a NetworkX DiGraph.

    Algorithm 1: Parse GV → NetworkX DiGraph
    """
    graphs = pydot.graph_from_dot_data(gv_content)
    if not graphs:
        raise ValueError("Failed to parse DOT content")

    dot_graph = graphs[0]
    G = nx.DiGraph()

    # Process all nodes (including from subgraphs)
    all_nodes = []
    all_edges = []

    def collect_from_graph(graph):
        """Recursively collect nodes and edges from graph and subgraphs."""
        for node in graph.get_nodes():
            all_nodes.append(node)
        for edge in graph.get_edges():
            all_edges.append(edge)
        for subgraph in graph.get_subgraphs():
            # Process all subgraphs including CLUSTER_XOR (we need nodes inside them)
            collect_from_graph(subgraph)

    collect_from_graph(dot_graph)

    # Process nodes
    for node in all_nodes:
        node_name = node.get_name().strip('"')

        # Skip special pydot nodes
        if node_name in ('node', 'edge', 'graph', ''):
            continue

        # Determine node type based on name and shape
        shape = node.get_shape()
        if shape:
            shape = shape.strip('"')

        node_type = 'task'
        label = node_name

        def extract_gateway_label(name, prefix):
            """Extract label from gateway name, removing the suffix code and number prefix."""
            raw_label = name[len(prefix):]
            # Remove trailing random suffix (e.g., _NEDB, _YT4H, _QT1I)
            if '_' in raw_label:
                parts = raw_label.rsplit('_', 1)
                # Check if last part looks like a suffix (4 alphanumeric chars)
                if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isalnum():
                    raw_label = parts[0]
            # Remove leading number prefix (e.g., "3. ", "12. ")
            raw_label = re.sub(r'^\d+\.\s*', '', raw_label)
            return raw_label

        if node_name == 'START_NODE':
            node_type = 'start'
            label = ''
        elif node_name == 'END_NODE':
            node_type = 'end'
            label = ''
        elif node_name.startswith('AND_SPLIT--'):
            node_type = 'and_split'
            label = extract_gateway_label(node_name, 'AND_SPLIT--')
        elif node_name.startswith('AND_JOIN--'):
            node_type = 'and_join'
            label = extract_gateway_label(node_name, 'AND_JOIN--')
        elif node_name.startswith('OR_SPLIT--'):
            node_type = 'or_split'
            label = extract_gateway_label(node_name, 'OR_SPLIT--')
        elif node_name.startswith('OR_JOIN--'):
            node_type = 'or_join'
            label = extract_gateway_label(node_name, 'OR_JOIN--')
        elif node_name.startswith('XOR_SPLIT--'):
            node_type = 'xor_split'
            label = extract_gateway_label(node_name, 'XOR_SPLIT--')
        elif node_name.startswith('XOR_JOIN--'):
            node_type = 'xor_join'
            label = extract_gateway_label(node_name, 'XOR_JOIN--')
        elif shape == 'box':
            node_type = 'task'
            label = node_name

        G.add_node(node_name, type=node_type, label=label)

    # Process edges
    for edge in all_edges:
        source = edge.get_source().strip('"')
        target = edge.get_destination().strip('"')

        # Get edge label if present
        edge_label = edge.get_label()
        if edge_label:
            edge_label = edge_label.strip('"')
        else:
            edge_label = ''

        G.add_edge(source, target, label=edge_label)

    return G


def compute_layout(G: nx.DiGraph) -> dict:
    """
    Compute layout coordinates for all nodes.

    Algorithm 2: Compute Layout
    Returns dict of node_id -> (x, y, width, height)
    """
    # Find start node
    start_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'start']
    start_node = start_nodes[0] if start_nodes else list(G.nodes())[0]

    # Compute levels (x-position) using longest path
    levels = {start_node: 0}

    # Use topological sort if possible, otherwise BFS
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Has cycles, use BFS order
        sorted_nodes = list(nx.bfs_tree(G, start_node).nodes()) if start_node in G else list(G.nodes())

    # Assign levels based on longest path from start
    for node in sorted_nodes:
        preds = list(G.predecessors(node))
        if preds:
            levels[node] = max(levels.get(p, 0) for p in preds) + 1
        elif node not in levels:
            levels[node] = 0

    # Feature 1: END_NODE level assignment (Algorithm 2, step 11)
    # Ensure END_NODE is always at the rightmost position
    max_level = max(levels.values()) if levels else 0
    for node in G.nodes():
        if G.nodes[node].get('type') == 'end':
            levels[node] = max_level + 1

    # Feature 2: Backward edge handling (Algorithm 2, step 17)
    # Detect and mark backward edges for special routing
    backward_edges = set()
    for u, v in G.edges():
        u_level = levels.get(u, 0)
        v_level = levels.get(v, 0)
        if u_level >= v_level:  # Backward edge (loop)
            backward_edges.add((u, v))

    # Store for use in generate_bpmn() edge routing
    G.graph['backward_edges'] = backward_edges

    # Compute y-positions using branch tracking
    y_positions = {}

    def get_node_dims(node):
        node_type = G.nodes[node].get('type', 'task')
        if node_type in ('start', 'end'):
            return EVENT_SIZE, EVENT_SIZE
        elif node_type in ('and_split', 'and_join', 'or_split', 'or_join', 'xor_split', 'xor_join'):
            return GATEWAY_SIZE, GATEWAY_SIZE
        else:
            return TASK_WIDTH, TASK_HEIGHT

    # Assign y-positions by traversing the graph and tracking branches
    def assign_y_positions(node, base_y, visited):
        if node in visited:
            return
        visited.add(node)

        y_positions[node] = base_y

        successors = list(G.successors(node))
        node_type = G.nodes[node].get('type', '')

        if len(successors) == 0:
            return
        elif len(successors) == 1:
            # Single successor - same y level
            assign_y_positions(successors[0], base_y, visited)
        else:
            # Multiple successors (split) - spread them out
            # Center the branches around the current y
            n = len(successors)
            total_height = (n - 1) * Y_SPACING
            start_y = base_y - total_height / 2

            for i, succ in enumerate(successors):
                succ_y = start_y + i * Y_SPACING
                assign_y_positions(succ, succ_y, visited)

    # Start from the start node
    assign_y_positions(start_node, START_Y + 200, set())

    # Handle any unvisited nodes
    for node in G.nodes():
        if node not in y_positions:
            # Find a predecessor with a y position
            preds = list(G.predecessors(node))
            if preds and preds[0] in y_positions:
                y_positions[node] = y_positions[preds[0]]
            else:
                y_positions[node] = START_Y

    # Feature 3: Gateway alignment adjustments (Algorithm 2, step 15)
    # Center join gateways between predecessors, split gateways between successors
    def is_join_gateway(node):
        return G.nodes[node].get('type', '') in ('and_join', 'or_join', 'xor_join')

    def is_split_gateway(node):
        return G.nodes[node].get('type', '') in ('and_split', 'or_split', 'xor_split')

    for node in G.nodes():
        if is_join_gateway(node):
            preds = list(G.predecessors(node))
            if len(preds) > 1:
                pred_ys = [y_positions.get(p, START_Y) for p in preds if p in y_positions]
                if pred_ys:
                    y_positions[node] = sum(pred_ys) / len(pred_ys)
        elif is_split_gateway(node):
            succs = list(G.successors(node))
            if len(succs) > 1:
                succ_ys = [y_positions.get(s, START_Y) for s in succs if s in y_positions]
                if succ_ys:
                    y_positions[node] = sum(succ_ys) / len(succ_ys)

    # Feature 4: AlternativeOverlapResolution (Algorithm 2, step 18)
    # Iterative multi-pass overlap resolution
    level_nodes = defaultdict(list)
    for node in G.nodes():
        level = levels.get(node, 0)
        level_nodes[level].append(node)

    MAX_ITERATIONS = 10

    for iteration in range(MAX_ITERATIONS):
        changes_made = False

        for level, nodes in level_nodes.items():
            if len(nodes) <= 1:
                continue

            # Sort by current y-position
            nodes_sorted = sorted(nodes, key=lambda n: y_positions.get(n, 0))

            for i in range(1, len(nodes_sorted)):
                prev_node = nodes_sorted[i-1]
                curr_node = nodes_sorted[i]
                prev_y = y_positions[prev_node]
                curr_y = y_positions[curr_node]

                _, prev_h = get_node_dims(prev_node)
                _, curr_h = get_node_dims(curr_node)
                min_gap = max(prev_h, curr_h, Y_SPACING)

                if curr_y < prev_y + min_gap:
                    y_positions[curr_node] = prev_y + min_gap
                    changes_made = True

        if not changes_made:
            break  # All overlaps resolved

    # Build final layout
    layout = {}
    for node in G.nodes():
        level = levels.get(node, 0)
        x = START_X + level * X_SPACING
        y = y_positions.get(node, START_Y)
        width, height = get_node_dims(node)
        layout[node] = (x, y, width, height)

    return layout


def sanitize_id(name: str) -> str:
    """Convert a node name to a valid BPMN ID."""
    # Replace problematic characters
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'N_' + sanitized
    return sanitized


def generate_bpmn(G: nx.DiGraph, layout: dict) -> str:
    """
    Generate BPMN 2.0 XML from graph and layout.

    Algorithm 3 & 4: Generate BPMN Process and Diagram
    """
    # BPMN namespaces
    NS = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'dc': 'http://www.omg.org/spec/DD/20100524/DC',
        'di': 'http://www.omg.org/spec/DD/20100524/DI',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }

    # Register namespaces
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)
    ET.register_namespace('', NS['bpmn'])

    # Create root definitions element
    root = ET.Element('{%s}definitions' % NS['bpmn'])
    root.set('id', 'Definitions_1')
    root.set('targetNamespace', 'http://bpmn.io/schema/bpmn')
    root.set('{%s}schemaLocation' % NS['xsi'],
             'http://www.omg.org/spec/BPMN/20100524/MODEL BPMN20.xsd')

    # Create process element
    process = ET.SubElement(root, '{%s}process' % NS['bpmn'])
    process.set('id', 'Process_1')
    process.set('isExecutable', 'false')

    # Track element IDs
    node_to_id = {}
    flow_elements = []
    flow_counter = [0]

    def get_flow_id():
        flow_counter[0] += 1
        return f'Flow_{flow_counter[0]}'

    # Create BPMN elements for each node
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'task')
        node_label = G.nodes[node].get('label', '')
        node_id = sanitize_id(node)
        node_to_id[node] = node_id

        if node_type == 'start':
            elem = ET.SubElement(process, '{%s}startEvent' % NS['bpmn'])
            elem.set('id', node_id)
            elem.set('name', 'Start')
        elif node_type == 'end':
            elem = ET.SubElement(process, '{%s}endEvent' % NS['bpmn'])
            elem.set('id', node_id)
            elem.set('name', 'End')
        elif node_type in ('and_split', 'and_join'):
            elem = ET.SubElement(process, '{%s}parallelGateway' % NS['bpmn'])
            elem.set('id', node_id)
            if node_label:
                elem.set('name', node_label)
        elif node_type in ('or_split', 'or_join'):
            elem = ET.SubElement(process, '{%s}inclusiveGateway' % NS['bpmn'])
            elem.set('id', node_id)
            if node_label:
                elem.set('name', node_label)
        elif node_type in ('xor_split', 'xor_join'):
            elem = ET.SubElement(process, '{%s}exclusiveGateway' % NS['bpmn'])
            elem.set('id', node_id)
            if node_label:
                elem.set('name', node_label)
        else:
            # Task
            elem = ET.SubElement(process, '{%s}task' % NS['bpmn'])
            elem.set('id', node_id)
            elem.set('name', node_label)

    # Create sequence flows
    for source, target, data in G.edges(data=True):
        flow_id = get_flow_id()
        source_id = node_to_id[source]
        target_id = node_to_id[target]
        edge_label = data.get('label', '')

        flow = ET.SubElement(process, '{%s}sequenceFlow' % NS['bpmn'])
        flow.set('id', flow_id)
        flow.set('sourceRef', source_id)
        flow.set('targetRef', target_id)

        # Add name attribute for edge label (visible on diagram)
        if edge_label:
            flow.set('name', edge_label)

        # Add condition expression for XOR edges with labels
        if edge_label:
            source_type = G.nodes[source].get('type', '')
            if source_type == 'xor_split':
                condition = ET.SubElement(flow, '{%s}conditionExpression' % NS['bpmn'])
                condition.set('{%s}type' % NS['xsi'], 'bpmn:tFormalExpression')
                condition.text = edge_label

        flow_elements.append((flow_id, source, target, edge_label))

    # Create BPMN diagram
    diagram = ET.SubElement(root, '{%s}BPMNDiagram' % NS['bpmndi'])
    diagram.set('id', 'BPMNDiagram_1')

    plane = ET.SubElement(diagram, '{%s}BPMNPlane' % NS['bpmndi'])
    plane.set('id', 'BPMNPlane_1')
    plane.set('bpmnElement', 'Process_1')

    # Create shapes for each node
    for node in G.nodes():
        node_id = node_to_id[node]
        x, y, width, height = layout[node]

        shape = ET.SubElement(plane, '{%s}BPMNShape' % NS['bpmndi'])
        shape.set('id', f'{node_id}_di')
        shape.set('bpmnElement', node_id)

        bounds = ET.SubElement(shape, '{%s}Bounds' % NS['dc'])
        bounds.set('x', str(int(x)))
        bounds.set('y', str(int(y)))
        bounds.set('width', str(int(width)))
        bounds.set('height', str(int(height)))

    # Create edges
    # Feature 2: Get backward edges for special routing
    backward_edges = G.graph.get('backward_edges', set())

    for flow_id, source, target, edge_label in flow_elements:
        source_x, source_y, source_w, source_h = layout[source]
        target_x, target_y, target_w, target_h = layout[target]

        edge = ET.SubElement(plane, '{%s}BPMNEdge' % NS['bpmndi'])
        edge.set('id', f'{flow_id}_di')
        edge.set('bpmnElement', flow_id)

        # Check if this is a backward edge (loop)
        if (source, target) in backward_edges:
            # Route backward edge below/above the flow with additional waypoints
            # Start from bottom of source, go down, across, up to left of target
            start_x = source_x + source_w / 2
            start_y = source_y + source_h  # Bottom of source
            end_x = target_x + target_w / 2
            end_y = target_y + target_h  # Bottom of target

            # Calculate offset for the loop (go below both nodes)
            loop_offset = 60
            mid_y = max(start_y, end_y) + loop_offset

            # Waypoint 1: Start (bottom center of source)
            wp1 = ET.SubElement(edge, '{%s}waypoint' % NS['di'])
            wp1.set('x', str(int(start_x)))
            wp1.set('y', str(int(start_y)))

            # Waypoint 2: Down from source
            wp2 = ET.SubElement(edge, '{%s}waypoint' % NS['di'])
            wp2.set('x', str(int(start_x)))
            wp2.set('y', str(int(mid_y)))

            # Waypoint 3: Across to target x
            wp3 = ET.SubElement(edge, '{%s}waypoint' % NS['di'])
            wp3.set('x', str(int(end_x)))
            wp3.set('y', str(int(mid_y)))

            # Waypoint 4: Up to target (bottom center of target)
            wp4 = ET.SubElement(edge, '{%s}waypoint' % NS['di'])
            wp4.set('x', str(int(end_x)))
            wp4.set('y', str(int(end_y)))
        else:
            # Normal forward edge: right side of source to left side of target
            start_x = source_x + source_w
            start_y = source_y + source_h / 2
            end_x = target_x
            end_y = target_y + target_h / 2

            # Start waypoint
            wp1 = ET.SubElement(edge, '{%s}waypoint' % NS['di'])
            wp1.set('x', str(int(start_x)))
            wp1.set('y', str(int(start_y)))

            # End waypoint
            wp2 = ET.SubElement(edge, '{%s}waypoint' % NS['di'])
            wp2.set('x', str(int(end_x)))
            wp2.set('y', str(int(end_y)))

        # Add label for edge if present
        if edge_label:
            label = ET.SubElement(edge, '{%s}BPMNLabel' % NS['bpmndi'])
            label_bounds = ET.SubElement(label, '{%s}Bounds' % NS['dc'])
            # Position label at midpoint of edge
            label_x = (start_x + end_x) / 2 - 50
            label_y = (start_y + end_y) / 2 - 10
            label_bounds.set('x', str(int(label_x)))
            label_bounds.set('y', str(int(label_y)))
            label_bounds.set('width', str(100))
            label_bounds.set('height', str(20))

    # Generate XML string with proper formatting
    ET.indent(root, space='  ')
    xml_str = ET.tostring(root, encoding='unicode', xml_declaration=False)

    # Add XML declaration
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str


def convert_file(gv_path: Path) -> bool:
    """Convert a single .gv file to .bpmn."""
    try:
        # Read GV content
        with open(gv_path, 'r', encoding='utf-8') as f:
            gv_content = f.read()

        # Parse to graph
        G = parse_gv_to_graph(gv_content)

        # Compute layout
        layout = compute_layout(G)

        # Generate BPMN
        bpmn_content = generate_bpmn(G, layout)

        # Write output
        bpmn_path = gv_path.with_suffix('.bpmn')
        with open(bpmn_path, 'w', encoding='utf-8') as f:
            f.write(bpmn_content)

        return True
    except Exception as e:
        print(f"Error converting {gv_path}: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert GV files to BPMN 2.0')
    parser.add_argument('path', nargs='?', default='.',
                        help='Directory to process (default: current directory)')
    parser.add_argument('--single', '-s', help='Convert a single file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    if args.single:
        # Convert single file
        gv_path = Path(args.single)
        if convert_file(gv_path):
            print(f"Converted: {gv_path} -> {gv_path.with_suffix('.bpmn')}")
        else:
            sys.exit(1)
    else:
        # Process directory recursively
        base_path = Path(args.path)
        gv_files = list(base_path.rglob('*.gv'))

        print(f"Found {len(gv_files)} .gv files")

        success = 0
        failed = 0

        for i, gv_path in enumerate(gv_files, 1):
            if args.verbose:
                print(f"[{i}/{len(gv_files)}] Converting {gv_path.name}...")

            if convert_file(gv_path):
                success += 1
            else:
                failed += 1

            # Progress indicator
            if i % 1000 == 0:
                print(f"Progress: {i}/{len(gv_files)} files processed")

        print(f"\nConversion complete:")
        print(f"  Success: {success}")
        print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
