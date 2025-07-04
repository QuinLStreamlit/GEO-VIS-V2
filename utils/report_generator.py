"""
Report Generator Module

This module provides functionality to generate professional PDF reports
from geotechnical analysis data and plots.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import io
from typing import Dict, List, Optional, Any
import base64

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def generate_timestamp():
    """Generate timestamp for report naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_report_styles():
    """Create custom styles for the report"""
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    # Custom heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    # Custom subheading style
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkblue
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'normal': styles['Normal'],
        'bullet': styles['Bullet']
    }


def add_plot_to_report(story, plot_buffer: io.BytesIO, title: str, width: float = 6*72):
    """Add a plot image to the report story"""
    if plot_buffer:
        try:
            plot_buffer.seek(0)
            if HAS_REPORTLAB:
                img = Image(plot_buffer, width=width, height=width*0.75)
            else:
                return
            img.hAlign = 'CENTER'
            
            styles = create_report_styles()
            story.append(Paragraph(title, styles['subheading']))
            story.append(Spacer(1, 12))
            story.append(img)
            story.append(Spacer(1, 20))
            
        except Exception as e:
            print(f"Error adding plot to report: {e}")


def generate_data_summary_table(filtered_data: pd.DataFrame) -> List:
    """Generate a data summary table for the report"""
    try:
        from .data_processing import get_test_availability
        
        test_availability = get_test_availability(filtered_data)
        
        # Create table data
        table_data = [['Test Type', 'Number of Records']]
        
        for test_type, count in test_availability.items():
            if count > 0:
                table_data.append([test_type, str(count)])
        
        # Add total
        table_data.append(['Total Records', str(len(filtered_data))])
        
        return table_data
        
    except Exception as e:
        print(f"Error generating data summary table: {e}")
        return [['Test Type', 'Number of Records'], ['Error', 'Unable to generate summary']]


def generate_geotechnical_investigation_report(filtered_data: pd.DataFrame, 
                                             generated_plots: Dict[str, io.BytesIO]) -> Optional[io.BytesIO]:
    """
    Generate a complete Geotechnical Investigation Report.
    
    Args:
        filtered_data: Laboratory data DataFrame
        generated_plots: Dictionary of plot names and buffers
        
    Returns:
        io.BytesIO: PDF report buffer
    """
    if not HAS_REPORTLAB:
        if HAS_STREAMLIT:
            st.error("ReportLab library not available. Cannot generate PDF reports.")
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = create_report_styles()
        
        # Title page
        story.append(Paragraph("Geotechnical Investigation Report", styles['title']))
        story.append(Spacer(1, 30))
        
        # Project information
        story.append(Paragraph("Project Information", styles['heading']))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", styles['normal']))
        story.append(Paragraph(f"Total Laboratory Records: {len(filtered_data)}", styles['normal']))
        
        # Add geology distribution if available
        if 'Geology_Orgin' in filtered_data.columns:
            geology_counts = filtered_data['Geology_Orgin'].value_counts()
            story.append(Paragraph("Geological Origins:", styles['normal']))
            for geology, count in geology_counts.items():
                story.append(Paragraph(f"• {geology}: {count} records", styles['bullet']))
        
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['heading']))
        story.append(Paragraph(
            "This report presents the results of laboratory testing conducted on soil and rock samples "
            "collected during the geotechnical investigation. The testing program included particle "
            "size distribution (PSD) analysis, Atterberg limits, Standard Penetration Test (SPT) data, "
            "unconfined compressive strength (UCS) testing, and Emerson class determination.",
            styles['normal']
        ))
        story.append(Spacer(1, 20))
        
        # Data Summary Table
        story.append(Paragraph("Test Data Summary", styles['heading']))
        table_data = generate_data_summary_table(filtered_data)
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # New page for plots
        story.append(PageBreak())
        
        # Add plots
        story.append(Paragraph("Laboratory Test Results", styles['heading']))
        
        plot_sections = {
            "PSD Analysis by Geology": "Particle Size Distribution Analysis",
            "Atterberg Classification Charts": "Atterberg Limits Classification",
            "SPT vs Depth (Cohesive)": "SPT Analysis - Cohesive Soils",
            "SPT vs Depth (Granular)": "SPT Analysis - Granular Soils", 
            "UCS vs Depth by Formation": "Unconfined Compressive Strength Analysis",
            "UCS vs Is50 Correlation": "UCS vs Point Load Index Correlation",
            "Emerson by Geological Origin": "Emerson Class Dispersivity Analysis",
            "Properties vs Chainage": "Spatial Variation of Properties",
            "Thickness Distribution": "Geological Formation Thickness Analysis"
        }
        
        for plot_name, plot_buffer in generated_plots.items():
            section_title = plot_sections.get(plot_name, plot_name)
            add_plot_to_report(story, plot_buffer, section_title)
        
        # Recommendations section
        story.append(PageBreak())
        story.append(Paragraph("Engineering Recommendations", styles['heading']))
        story.append(Paragraph(
            "Based on the laboratory test results presented in this report, the following "
            "preliminary recommendations are provided:",
            styles['normal']
        ))
        story.append(Spacer(1, 12))
        
        # Add specific recommendations based on available data
        recommendations = []
        
        try:
            from .data_processing import get_test_availability
            test_availability = get_test_availability(filtered_data)
            
            if test_availability.get('PSD', 0) > 0:
                recommendations.append(
                    "Particle size distribution analysis indicates the presence of varied soil types. "
                    "Foundation design should consider the heterogeneous nature of subsurface conditions."
                )
            
            if test_availability.get('Atterberg', 0) > 0:
                recommendations.append(
                    "Atterberg limits testing provides classification of fine-grained soils. "
                    "Consider plasticity characteristics in earthwork and foundation design."
                )
            
            if test_availability.get('UCS', 0) > 0:
                recommendations.append(
                    "Unconfined compressive strength testing indicates rock strength variations. "
                    "Excavation methods and foundation bearing capacity should be evaluated accordingly."
                )
            
            if test_availability.get('Emerson', 0) > 0:
                recommendations.append(
                    "Emerson class testing indicates dispersive potential of soils. "
                    "Special considerations for erosion control and earthwork may be required."
                )
            
        except:
            recommendations.append(
                "Detailed engineering analysis should be conducted based on the laboratory test results "
                "to develop site-specific design recommendations."
            )
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['normal']))
            story.append(Spacer(1, 8))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Generated by Geotechnical Data Analysis Tool", styles['normal']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        if HAS_STREAMLIT:
            st.error(f"Error generating PDF report: {str(e)}")
        print(f"Error generating PDF report: {str(e)}")
        return None


def generate_foundation_design_summary(filtered_data: pd.DataFrame, 
                                     generated_plots: Dict[str, io.BytesIO]) -> Optional[io.BytesIO]:
    """
    Generate a Foundation Design Summary Report.
    """
    if not HAS_REPORTLAB:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = create_report_styles()
        
        # Title
        story.append(Paragraph("Foundation Design Summary", styles['title']))
        story.append(Spacer(1, 30))
        
        # Introduction
        story.append(Paragraph("Introduction", styles['heading']))
        story.append(Paragraph(
            "This report summarizes the geotechnical data relevant for foundation design, "
            "including SPT analysis, unconfined compressive strength testing, and bearing capacity considerations.",
            styles['normal']
        ))
        story.append(Spacer(1, 20))
        
        # Add relevant plots
        foundation_plots = ["SPT vs Depth (Cohesive)", "SPT vs Depth (Granular)", 
                          "UCS vs Depth by Formation", "UCS vs Is50 Correlation"]
        
        for plot_name in foundation_plots:
            if plot_name in generated_plots:
                add_plot_to_report(story, generated_plots[plot_name], plot_name)
        
        # Foundation recommendations
        story.append(Paragraph("Foundation Design Considerations", styles['heading']))
        story.append(Paragraph(
            "Based on the available geotechnical data, preliminary foundation design parameters "
            "can be estimated. Detailed analysis should be conducted by a qualified geotechnical engineer.",
            styles['normal']
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating foundation design summary: {str(e)}")
        return None


def generate_material_characterization_report(filtered_data: pd.DataFrame, 
                                            generated_plots: Dict[str, io.BytesIO]) -> Optional[io.BytesIO]:
    """
    Generate a Material Characterization Report.
    """
    if not HAS_REPORTLAB:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = create_report_styles()
        
        # Title
        story.append(Paragraph("Material Characterization Report", styles['title']))
        story.append(Spacer(1, 30))
        
        # Introduction
        story.append(Paragraph("Material Properties Overview", styles['heading']))
        story.append(Paragraph(
            "This report presents detailed material characterization based on laboratory testing "
            "including particle size distribution, Atterberg limits, and dispersivity analysis.",
            styles['normal']
        ))
        story.append(Spacer(1, 20))
        
        # Add relevant plots
        material_plots = ["PSD Analysis by Geology", "Atterberg Classification Charts", 
                         "Emerson by Geological Origin"]
        
        for plot_name in material_plots:
            if plot_name in generated_plots:
                add_plot_to_report(story, generated_plots[plot_name], plot_name)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating material characterization report: {str(e)}")
        return None


def generate_custom_report(filtered_data: pd.DataFrame, 
                         generated_plots: Dict[str, io.BytesIO],
                         title: str,
                         author: str,
                         selected_sections: List[str],
                         include_recommendations: bool,
                         include_data_tables: bool) -> Optional[io.BytesIO]:
    """
    Generate a custom report based on user selections.
    
    Args:
        filtered_data: Laboratory data DataFrame
        generated_plots: Dictionary of plot names and buffers
        title: Custom report title
        author: Author/company name
        selected_sections: List of sections to include
        include_recommendations: Whether to include recommendations
        include_data_tables: Whether to include data tables
        
    Returns:
        io.BytesIO: PDF report buffer
    """
    if not HAS_REPORTLAB:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = create_report_styles()
        
        # Title page
        story.append(Paragraph(title, styles['title']))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Prepared by: {author}", styles['normal']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", styles['normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary (if selected)
        if "Executive Summary" in selected_sections:
            story.append(Paragraph("Executive Summary", styles['heading']))
            story.append(Paragraph(
                f"This report presents the analysis of {len(filtered_data)} laboratory test records "
                f"collected during the geotechnical investigation. The analysis includes selected "
                f"geotechnical parameters and engineering assessments based on available data.",
                styles['normal']
            ))
            story.append(Spacer(1, 20))
        
        # Data Overview (if selected)
        if "Data Overview" in selected_sections:
            story.append(Paragraph("Data Overview", styles['heading']))
            story.append(Paragraph(f"Total Laboratory Records: {len(filtered_data)}", styles['normal']))
            
            # Add geology distribution if available
            if 'Geology_Orgin' in filtered_data.columns:
                geology_counts = filtered_data['Geology_Orgin'].value_counts()
                story.append(Paragraph("Geological Origins:", styles['normal']))
                for geology, count in geology_counts.items():
                    story.append(Paragraph(f"• {geology}: {count} records", styles['bullet']))
            
            # Data summary table (if selected)
            if include_data_tables:
                story.append(Spacer(1, 15))
                story.append(Paragraph("Test Data Summary", styles['subheading']))
                table_data = generate_data_summary_table(filtered_data)
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
            
            story.append(Spacer(1, 20))
        
        # Analysis sections
        analysis_sections = {
            "PSD Analysis": "Particle Size Distribution Analysis",
            "Atterberg Limits": "Atterberg Limits Analysis", 
            "SPT Analysis": "Standard Penetration Test Analysis",
            "Rock Strength Analysis": "Rock Strength Analysis",
            "Emerson Dispersivity": "Emerson Dispersivity Analysis",
            "Spatial Analysis": "Spatial Analysis"
        }
        
        for section in selected_sections:
            if section in analysis_sections:
                story.append(Paragraph(analysis_sections[section], styles['heading']))
                story.append(Paragraph(
                    f"This section presents the {section.lower()} results based on laboratory testing. "
                    f"The analysis provides insights into material properties and engineering characteristics.",
                    styles['normal']
                ))
                story.append(Spacer(1, 20))
        
        # Add plots
        if generated_plots:
            story.append(PageBreak())
            story.append(Paragraph("Test Results and Analysis", styles['heading']))
            
            for plot_name, plot_buffer in generated_plots.items():
                add_plot_to_report(story, plot_buffer, plot_name)
        
        # Engineering Recommendations (if selected)
        if include_recommendations and "Engineering Recommendations" in selected_sections:
            story.append(PageBreak())
            story.append(Paragraph("Engineering Recommendations", styles['heading']))
            story.append(Paragraph(
                "Based on the laboratory test results and analysis presented in this report, "
                "the following preliminary engineering recommendations are provided:",
                styles['normal']
            ))
            story.append(Spacer(1, 12))
            
            # General recommendations
            recommendations = [
                "Detailed site-specific analysis should be conducted by a qualified geotechnical engineer.",
                "Foundation design should consider the variability in subsurface conditions identified through testing.",
                "Construction activities should follow appropriate geotechnical guidelines and monitoring protocols.",
                "Additional testing may be recommended based on specific project requirements and site conditions."
            ]
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", styles['normal']))
                story.append(Spacer(1, 8))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Generated by Geotechnical Data Analysis Tool", styles['normal']))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        print(f"Error generating custom report: {str(e)}")
        return None