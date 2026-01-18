#!/usr/bin/env python3
"""
Generate methodology diagrams for Chapter 3
"""

import sys
import os
# Fix Unicode encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# ============================================================================
# Figure 1: Model Architecture Diagram
# ============================================================================
print("Generating Figure 1: Model Architecture Diagram...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_input = '#E8F4F8'
color_gat = '#FFE5B4'
color_codebert = '#D4EDDA'
color_fusion = '#F8D7DA'
color_output = '#E2E3E5'

# Input Layer
input_box = FancyBboxPatch((0.5, 4), 1.5, 2, boxstyle="round,pad=0.1", 
                           facecolor=color_input, edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 5.5, 'Source\nCode', ha='center', va='center', fontsize=12, weight='bold')
ax.text(1.25, 4.5, '(C/C++)', ha='center', va='center', fontsize=10)

# Split into two paths
ax.arrow(2, 5, 1, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)

# GAT Path
gat_box1 = FancyBboxPatch((3.2, 6.5), 1.5, 1, boxstyle="round,pad=0.1",
                         facecolor=color_gat, edgecolor='black', linewidth=1.5)
ax.add_patch(gat_box1)
ax.text(3.95, 7, 'AST\nParser', ha='center', va='center', fontsize=10, weight='bold')

ax.arrow(4.7, 7, 0.8, 0, head_width=0.1, head_length=0.08, fc='black', ec='black')

gat_box2 = FancyBboxPatch((5.5, 6.5), 1.5, 1, boxstyle="round,pad=0.1",
                         facecolor=color_gat, edgecolor='black', linewidth=1.5)
ax.add_patch(gat_box2)
ax.text(6.25, 7, 'GAT\nEncoder', ha='center', va='center', fontsize=10, weight='bold')
ax.text(6.25, 6.6, '256-dim', ha='center', va='center', fontsize=9)

# CodeBERT Path
codebert_box1 = FancyBboxPatch((3.2, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor=color_codebert, edgecolor='black', linewidth=1.5)
ax.add_patch(codebert_box1)
ax.text(3.95, 4, 'CodeBERT\nTokenizer', ha='center', va='center', fontsize=10, weight='bold')

ax.arrow(4.7, 4, 0.8, 0, head_width=0.1, head_length=0.08, fc='black', ec='black')

codebert_box2 = FancyBboxPatch((5.5, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor=color_codebert, edgecolor='black', linewidth=1.5)
ax.add_patch(codebert_box2)
ax.text(6.25, 4, 'CodeBERT\nEncoder', ha='center', va='center', fontsize=10, weight='bold')
ax.text(6.25, 3.6, '768-dim', ha='center', va='center', fontsize=9)

# Merge arrows
ax.arrow(7, 7, 0.5, -1, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
ax.arrow(7, 4, 0.5, 1, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)

# Fusion Layer
fusion_box = FancyBboxPatch((7.8, 4.5), 1.5, 2, boxstyle="round,pad=0.1",
                            facecolor=color_fusion, edgecolor='black', linewidth=2)
ax.add_patch(fusion_box)
ax.text(8.55, 6.2, 'Fusion\nLayer', ha='center', va='center', fontsize=11, weight='bold')
ax.text(8.55, 5.5, 'Concat + MLP', ha='center', va='center', fontsize=9)
ax.text(8.55, 5, '1024→512→256', ha='center', va='center', fontsize=9)

ax.arrow(9.3, 5.5, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)

# Output Layer
output_box = FancyBboxPatch((9.8, 4.5), 1.5, 2, boxstyle="round,pad=0.1",
                           facecolor=color_output, edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(10.55, 5.5, 'Binary\nClassification', ha='center', va='center', fontsize=11, weight='bold')
ax.text(10.55, 4.8, 'Vulnerable/\nNon-Vulnerable', ha='center', va='center', fontsize=9)

# Title
ax.text(5.5, 9.5, 'Hybrid GNN-Transformer Model Architecture', 
        ha='center', va='center', fontsize=16, weight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=color_gat, edgecolor='black', label='GAT Encoder Path'),
    mpatches.Patch(facecolor=color_codebert, edgecolor='black', label='CodeBERT Encoder Path'),
    mpatches.Patch(facecolor=color_fusion, edgecolor='black', label='Fusion Layer'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('visualizations/methodology_1_model_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: methodology_1_model_architecture.png")

# ============================================================================
# Figure 2: AST Graph Example
# ============================================================================
print("Generating Figure 2: AST Graph Example...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Left: Code Example
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

code_text = """int vulnerable_func(int x) {
    if (x > 0) {
        return x * 2;
    }
    return 0;
}"""

ax1.text(5, 8, 'Source Code Example', ha='center', va='center', 
         fontsize=14, weight='bold')
ax1.add_patch(Rectangle((1, 2), 8, 5, fill=False, edgecolor='black', linewidth=2))
ax1.text(5, 4.5, code_text, ha='center', va='center', 
         fontsize=11, family='Courier New', 
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))

# Right: AST Visualization
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

ax2.text(5, 9.5, 'Abstract Syntax Tree (AST)', ha='center', va='center',
         fontsize=14, weight='bold')

# Draw AST nodes
nodes = {
    'func': (5, 8),
    'if': (3, 6.5),
    'return1': (7, 6.5),
    'return2': (5, 5),
    'condition': (2, 5),
    'mult': (6, 5),
    'x': (1.5, 3.5),
    '0': (2.5, 3.5),
    'x2': (5.5, 3.5),
    '2': (6.5, 3.5),
    '0_ret': (5, 3.5),
}

# Draw edges
edges = [
    ('func', 'if'), ('func', 'return2'),
    ('if', 'condition'), ('if', 'return1'),
    ('condition', 'x'), ('condition', '0'),
    ('return1', 'mult'),
    ('mult', 'x2'), ('mult', '2'),
    ('return2', '0_ret'),
]

for (n1, n2) in edges:
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]
    ax2.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.6)

# Draw nodes
for name, (x, y) in nodes.items():
    if name == 'func':
        color = '#FFE5B4'
        size = 0.4
    elif name in ['if', 'return1', 'return2']:
        color = '#D4EDDA'
        size = 0.3
    else:
        color = '#E8F4F8'
        size = 0.25
    
    circle = Circle((x, y), size, facecolor=color, edgecolor='black', linewidth=1.5)
    ax2.add_patch(circle)
    ax2.text(x, y, name, ha='center', va='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('visualizations/methodology_2_ast_example.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: methodology_2_ast_example.png")

# ============================================================================
# Figure 3: Data Flow Pipeline
# ============================================================================
print("Generating Figure 3: Data Flow Pipeline...")

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Pipeline stages
stages = [
    ('Raw\nDataset\n(CSV)', 1.5, 3, '#E8F4F8'),
    ('Data\nPreprocessing', 4, 3, '#FFF4E6'),
    ('Graph\nConstruction\n(AST)', 6.5, 3, '#E1F5FE'),
    ('Model\nTraining', 9, 3, '#F3E5F5'),
    ('Model\nEvaluation', 11.5, 3, '#E8F5E9'),
    ('Results &\nMetrics', 14, 3, '#FFEBEE'),
]

# Draw stages
for i, (label, x, y, color) in enumerate(stages):
    box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, boxstyle="round,pad=0.1",
                        facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows
    if i < len(stages) - 1:
        ax.arrow(x+0.8, y, 0.9, 0, head_width=0.2, head_length=0.15, 
                fc='black', ec='black', linewidth=2)

# Add details below
details = [
    '21,854 samples\n80-20 split',
    'Tokenization\nAST Parsing',
    'GAT + CodeBERT\nHybrid Model',
    '25 epochs\nAdamW optimizer',
    'Accuracy, F1\nPrecision, Recall',
    'Visualizations\nPerformance Analysis',
]

for i, (label, x, y, color) in enumerate(stages):
    ax.text(x, y-1.5, details[i], ha='center', va='center', 
           fontsize=9, style='italic', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

# Title
ax.text(8, 5.2, 'End-to-End Training Pipeline', 
        ha='center', va='center', fontsize=16, weight='bold')

plt.tight_layout()
plt.savefig('visualizations/methodology_3_data_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: methodology_3_data_pipeline.png")

print("\n[OK] All methodology diagrams generated successfully!")
print("  Files saved in: visualizations/")
print("  - methodology_1_model_architecture.png")
print("  - methodology_2_ast_example.png")
print("  - methodology_3_data_pipeline.png")
