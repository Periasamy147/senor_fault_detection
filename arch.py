import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Create a figure and axis with a larger size for detailed layout
plt.figure(figsize=(12, 8), facecolor='white')
ax = plt.gca()
ax.axis('off')  # Hide axes

# Create a directed graph using networkx
G = nx.DiGraph()

# Define layer details with detailed annotations
layers = [
    {"name": "Input Layer", "details": "2 features, sequence length=10", "pos": (0, 5)},
    {"name": "BiLSTM Layer 1", "details": "64 units, Dropout=0.2", "pos": (0, 4)},
    {"name": "BiLSTM Layer 2", "details": "32 units, Dropout=0.2", "pos": (0, 3)},
    {"name": "Dense Layer", "details": "16 units, ReLU activation", "pos": (0, 2)},
    {"name": "Output Layer (Fault)", "details": "1 unit, Sigmoid activation", "pos": (-1, 1)},
    {"name": "Output Layer (Imputation)", "details": "1 unit, Linear activation", "pos": (1, 1)}
]

# Add nodes to the graph with detailed labels
for layer in layers:
    G.add_node(layer["name"], details=layer["details"], pos=layer["pos"])

# Add edges to represent data flow
edges = [
    ("Input Layer", "BiLSTM Layer 1"),
    ("BiLSTM Layer 1", "BiLSTM Layer 2"),
    ("BiLSTM Layer 2", "Dense Layer"),
    ("Dense Layer", "Output Layer (Fault)"),
    ("Dense Layer", "Output Layer (Imputation)")
]
G.add_edges_from(edges)

# Get positions for all nodes
pos = nx.get_node_attributes(G, 'pos')

# Draw nodes with detailed styling
node_colors = ['lightgreen', 'lightblue', 'lightblue', 'lightyellow', 'lightcoral', 'lightcoral']
node_sizes = [4000] * len(layers)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

# Draw labels with detailed text
for node, (x, y) in pos.items():
    details = G.nodes[node]["details"]
    ax.text(x, y, f"{node}\n{details}", ha='center', va='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), 
            weight='bold' if "Output" in node else 'normal')

# Customize arrows with detailed styling
arrow_style = FancyArrowPatch
for edge in G.edges():
    src, tgt = edge
    src_pos = pos[src]
    tgt_pos = pos[tgt]
    arrow = FancyArrowPatch(src_pos, tgt_pos, arrowstyle="->", mutation_scale=20, 
                           lw=2, color='black', connectionstyle="arc3,rad=0.1")
    ax.add_patch(arrow)

# Add title and additional annotations
ax.text(0, 6, "BiLSTM Model Architecture for Sensor Fault Detection", 
        ha='center', fontsize=14, weight='bold', bbox=dict(facecolor='white', alpha=0.9))

# Add legend or notes
ax.text(-5, 0, "Notes:\n- BiLSTM layers capture bidirectional temporal dependencies.\n- Dropout prevents overfitting.\n- Two outputs for fault detection and imputation.", 
        ha='left', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

# Adjust layout and add grid for alignment (optional)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save the detailed figure with high resolution
plt.savefig('bilstm_architecture_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Detailed BiLSTM architecture image saved as 'bilstm_architecture_detailed.png'")