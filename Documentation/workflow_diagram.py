"""
Data Flow Workflow Diagram Generator
Creates a visual representation of the data-to-plot workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_workflow_diagram():
    """Create a comprehensive data flow diagram"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#FFE4B5',      # Moccasin - Data input
        'process': '#87CEEB',    # Sky blue - Processing
        'analysis': '#98FB98',   # Pale green - Analysis
        'ui': '#DDA0DD',         # Plum - UI layer
        'output': '#F0E68C'      # Khaki - Output
    }
    
    # Helper function to create boxes
    def create_box(ax, x, y, width, height, text, color, text_size=10):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=text_size, 
                weight='bold', wrap=True)
        return box
    
    # Helper function to create arrows
    def create_arrow(ax, start_x, start_y, end_x, end_y, text=''):
        arrow = ConnectionPatch(
            (start_x, start_y), (end_x, end_y),
            "data", "data",
            arrowstyle="->",
            shrinkA=5, shrinkB=5,
            mutation_scale=20,
            fc="black", ec="black",
            linewidth=2
        )
        ax.add_patch(arrow)
        
        # Add text along arrow if provided
        if text:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=8, 
                   style='italic', ha='center')
    
    # Title
    ax.text(5, 19.5, 'Geotechnical Data Analysis: Data-to-Plot Workflow', 
            ha='center', va='center', fontsize=18, weight='bold')
    
    # Stage 1: Data Input
    create_box(ax, 1, 17.5, 8, 1.5, 
               'STAGE 1: DATA INPUT\n' +
               'Lab_summary_final.xlsx (2,459 × 167)\n' +
               'BH_Interpretation.xlsx (1,893 × 7)',
               colors['input'], 12)
    
    create_arrow(ax, 5, 17.5, 5, 16.5)
    
    # Stage 2: Data Loading & Validation
    create_box(ax, 0.5, 15, 4, 1.5,
               'Data Loading\n& Validation\n' +
               '(data_processing.py)\n' +
               '• Structure validation\n' +
               '• Type detection\n' +
               '• Test availability',
               colors['process'], 9)
    
    create_box(ax, 5.5, 15, 4, 1.5,
               'Dynamic Column\nExtraction\n' +
               '(extract_test_columns)\n' +
               '• Find test boundaries\n' +
               '• Extract relevant columns\n' +
               '• Map to test types',
               colors['process'], 9)
    
    create_arrow(ax, 2.5, 15, 2.5, 14)
    create_arrow(ax, 7.5, 15, 7.5, 14)
    
    # Stage 3: Test-Specific Processing (Multiple branches)
    y_pos = 12.5
    test_boxes = [
        ('PSD Analysis\n• Wide→Long format\n• Sieve size parsing\n• Geological grouping', 0.2),
        ('Atterberg Analysis\n• Column identification\n• Data validation\n• Classification logic', 2.2),
        ('UCS Analysis\n• Depth relationships\n• Statistical analysis\n• Strength classification', 4.2),
        ('Spatial Analysis\n• Property selection\n• Zone classification\n• Chainage filtering', 6.2),
        ('Thickness Analysis\n• Formation grouping\n• Proportion calculation\n• Category distribution', 8.2)
    ]
    
    for i, (text, x_pos) in enumerate(test_boxes):
        create_box(ax, x_pos, y_pos, 1.6, 1.5, text, colors['analysis'], 8)
        create_arrow(ax, x_pos + 0.8, y_pos + 1.5, x_pos + 0.8, y_pos + 2)
    
    # Convergence arrow
    create_arrow(ax, 5, 12.5, 5, 11.5)
    
    # Stage 4: Parameter Standardization
    create_box(ax, 2, 10, 6, 1.5,
               'STAGE 4: PARAMETER STANDARDIZATION\n' +
               '(plot_defaults.py)\n' +
               '• Professional standards • Color schemes\n' +
               '• Font sizing • Default parameters',
               colors['process'], 10)
    
    create_arrow(ax, 5, 10, 5, 9)
    
    # Stage 5: UI Parameter Collection
    create_box(ax, 1, 7.5, 8, 1.5,
               'STAGE 5: UI PARAMETER COLLECTION\n' +
               'Standardized 5×5 Parameter Grid\n' +
               'Row 1: Data Options | Row 2: Plot Config | Row 3: Visual Style\n' +
               'Row 4: Font Styling | Row 5: Advanced Options',
               colors['ui'], 10)
    
    create_arrow(ax, 5, 7.5, 5, 6.5)
    
    # Stage 6: Plot Generation Pipeline
    create_box(ax, 0.5, 5, 2, 1.5,
               'Function Call\nBridge\n' +
               '(plotting_utils.py)\n' +
               '• Backend setup\n' +
               '• Figure capture',
               colors['process'], 9)
    
    create_box(ax, 3, 5, 2, 1.5,
               'Original Plotting\nFunctions\n' +
               '(Functions/ folder)\n' +
               '• 16 specialized plots\n' +
               '• Jupyter notebook code',
               colors['analysis'], 9)
    
    create_box(ax, 5.5, 5, 2, 1.5,
               'Matplotlib\nGeneration\n' +
               '• Professional figures\n' +
               '• High-quality output\n' +
               '• Industry standards',
               colors['output'], 9)
    
    create_box(ax, 8, 5, 1.5, 1.5,
               'Streamlit\nDisplay\n' +
               '• Figure capture\n' +
               '• Size control\n' +
               '• User display',
               colors['ui'], 9)
    
    # Arrows for pipeline
    create_arrow(ax, 1.5, 6.5, 1.5, 6.5)
    create_arrow(ax, 2.5, 5.75, 3, 5.75)
    create_arrow(ax, 5, 5.75, 5.5, 5.75)
    create_arrow(ax, 7.5, 5.75, 8, 5.75)
    
    # Stage 7: Output
    create_box(ax, 1, 2.5, 3, 1.5,
               'Plot Display\n& Download\n' +
               '• Streamlit interface\n' +
               '• Download buttons\n' +
               '• Size control',
               colors['output'], 10)
    
    create_box(ax, 6, 2.5, 3, 1.5,
               'Dashboard Storage\n& Gallery\n' +
               '• Plot caching\n' +
               '• Gallery views\n' +
               '• Batch export',
               colors['output'], 10)
    
    create_arrow(ax, 6.25, 5, 2.5, 4)
    create_arrow(ax, 8.25, 5, 7.5, 4)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Data Input'),
        mpatches.Patch(color=colors['process'], label='Data Processing'),
        mpatches.Patch(color=colors['analysis'], label='Analysis Modules'),
        mpatches.Patch(color=colors['ui'], label='User Interface'),
        mpatches.Patch(color=colors['output'], label='Output Generation')
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.01), ncol=5, fontsize=12)
    
    # Data flow summary on the right
    ax.text(9.5, 15, 'KEY TRANSFORMATIONS', rotation=90, 
            ha='center', va='center', fontsize=14, weight='bold')
    
    transformations = [
        '2,459 × 167 → Validated DataFrame',
        'Wide Format → Long Format (PSD)',
        'Raw Values → Classifications',
        'Multiple Tests → Unified Interface',
        'User Input → Plot Parameters',
        'Data + Parameters → Professional Plots'
    ]
    
    for i, transform in enumerate(transformations):
        ax.text(9.2, 14 - i*2, f'• {transform}', fontsize=9, 
                ha='left', va='center')
    
    plt.tight_layout()
    return fig

def create_architecture_diagram():
    """Create a system architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'ui': '#FFB6C1',         # Light pink - UI layer
        'utils': '#98FB98',      # Pale green - Utilities
        'functions': '#87CEEB',  # Sky blue - Core functions
        'data': '#F0E68C'        # Khaki - Data layer
    }
    
    # Helper function to create rounded rectangles
    def create_module_box(ax, x, y, width, height, title, items, color):
        # Main box
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Title
        ax.text(x + width/2, y + height - 0.3, title, 
                ha='center', va='center', fontsize=12, weight='bold')
        
        # Items
        for i, item in enumerate(items):
            ax.text(x + 0.1, y + height - 0.8 - i*0.3, f'• {item}', 
                    ha='left', va='center', fontsize=9)
    
    # Title
    ax.text(7, 9.5, 'System Architecture: Component Relationships', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # UI Layer
    create_module_box(ax, 1, 7.5, 12, 1.5, 'USER INTERFACE LAYER',
                     ['main_app.py - Main Streamlit application',
                      'Sidebar controls, Tab navigation, File upload interface'], 
                     colors['ui'])
    
    # Utils Layer
    utils_modules = [
        'data_processing.py - Data loading & validation',
        'plotting_utils.py - Streamlit-matplotlib bridge',
        'plot_defaults.py - Parameter standardization',
        '*_analysis.py - Test-specific processing'
    ]
    create_module_box(ax, 1, 5, 6, 2, 'UTILITIES LAYER (utils/)', 
                     utils_modules, colors['utils'])
    
    # Functions Layer
    functions_modules = [
        'plot_psd.py - Particle size curves',
        'plot_atterberg_chart.py - Plasticity charts',
        'plot_UCS_vs_depth.py - Strength profiles',
        'plot_by_chainage.py - Spatial analysis',
        '12 other specialized plotting functions'
    ]
    create_module_box(ax, 8, 5, 5, 2, 'PLOTTING ENGINE (Functions/)', 
                     functions_modules, colors['functions'])
    
    # Data Layer
    data_items = [
        'Lab_summary_final.xlsx - Main data',
        'BH_Interpretation.xlsx - Thickness data',
        'Output/ - Generated plots',
        'Session state - Runtime data'
    ]
    create_module_box(ax, 4, 2, 6, 2, 'DATA LAYER', 
                     data_items, colors['data'])
    
    # Arrows showing relationships
    def create_relationship_arrow(start_x, start_y, end_x, end_y, label):
        arrow = ConnectionPatch(
            (start_x, start_y), (end_x, end_y),
            "data", "data",
            arrowstyle="<->",
            shrinkA=5, shrinkB=5,
            mutation_scale=15,
            fc="red", ec="red",
            linewidth=2
        )
        ax.add_patch(arrow)
        
        # Label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.2, label, fontsize=8, 
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Relationships
    create_relationship_arrow(4, 7.5, 4, 7, 'User Input')
    create_relationship_arrow(10.5, 7.5, 10.5, 7, 'Display Output')
    create_relationship_arrow(7, 6, 8, 6, 'Function Calls')
    create_relationship_arrow(4, 5, 7, 4, 'Data Flow')
    create_relationship_arrow(10.5, 5, 7, 4, 'Results')
    
    # Add design principles box
    principles_text = [
        'DESIGN PRINCIPLES:',
        '• Separation of Concerns',
        '• Golden Standard Fidelity',
        '• Modular Architecture',
        '• Professional Output Quality'
    ]
    
    for i, text in enumerate(principles_text):
        weight = 'bold' if i == 0 else 'normal'
        ax.text(0.5, 1.5 - i*0.2, text, fontsize=10, weight=weight)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Generate workflow diagram
    print("Generating data flow workflow diagram...")
    workflow_fig = create_workflow_diagram()
    workflow_fig.savefig('Documentation/data_flow_workflow.png', 
                        dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
    print("✓ Workflow diagram saved: Documentation/data_flow_workflow.png")
    
    # Generate architecture diagram
    print("Generating system architecture diagram...")
    arch_fig = create_architecture_diagram()
    arch_fig.savefig('Documentation/system_architecture.png', 
                     dpi=300, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
    print("✓ Architecture diagram saved: Documentation/system_architecture.png")
    
    plt.close('all')
    print("\n📋 Documentation files created:")
    print("• PLOT_WORKFLOW_DOCUMENTATION.md - Comprehensive workflow guide")
    print("• data_flow_workflow.png - Visual data flow diagram")
    print("• system_architecture.png - System component diagram")
    print("\n🎯 All files located in Documentation/ folder")