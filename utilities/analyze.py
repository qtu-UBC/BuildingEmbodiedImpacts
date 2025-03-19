from config import config

"""
This module contains functions for analyzing and visualizing building material data.
"""

"""
================
Import libraries
================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import os

def create_material_intensity_boxplots(
    df: pd.DataFrame,
    material_col: str,
    group_by: str = "Occupation",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    palette: str = "Set2"
) -> plt.Figure:
    """
    Creates a panel of boxplots showing the intensity (kg/m²) of a specific material
    across different categories (e.g., building occupations).
    
    Args:
        df (pd.DataFrame): DataFrame containing the building data with material intensity values
                          already calculated in kg/m²
        material_col (str): Column name of the material intensity to analyze
        group_by (str, optional): Column to group the data by. Defaults to "Occupation".
        figsize (tuple, optional): Figure size as (width, height). Defaults to (12, 8).
        output_path (str, optional): Path to save the figure. If None, figure is saved to default output folder.
        title (str, optional): Custom title for the plot. If None, a default title is generated.
        palette (str, optional): Color palette for the boxplots. Defaults to "Set2".
    
    Returns:
        plt.Figure: The created figure object
    
    Example:
        >>> create_material_intensity_boxplots(
        ...     df=intensity_df,
        ...     material_col="copper_intensity",
        ...     group_by="Occupation",
        ...     output_path="output/copper_intensity_by_occupation.png"
        ... )
    """
    # Work with a copy of the dataframe
    df = df.copy()
    
    # Ensure the material column exists
    if material_col not in df.columns:
        raise ValueError(f"Material column '{material_col}' not found in DataFrame")
    
    # Ensure the grouping column exists
    if group_by not in df.columns:
        raise ValueError(f"Group column '{group_by}' not found in DataFrame")
    
    # Remove rows with NaN or infinite values
    df = df[np.isfinite(df[material_col])]
    
    # Add more detailed debug prints
    print(f"Debug: DataFrame shape: {df.shape}")
    print(f"Debug: Number of finite intensity values: {np.sum(np.isfinite(df[material_col]))}")
    print(f"Debug: Unique {group_by} values: {df[group_by].unique()}")
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    print("Debug: Figure created")
    
    # Create the boxplot
    sns.boxplot(
        x=group_by,
        y=material_col,
        data=df,
        ax=ax,
        palette=palette
    )
    
    # Add individual data points for better visualization
    sns.stripplot(
        x=group_by,
        y=material_col,
        data=df,
        ax=ax,
        color='black',
        alpha=0.5,
        size=4,
        jitter=True
    )
    
    # Set title and labels
    material_name = material_col.replace("MAT_", "").title()
    if title is None:
        title = f"{material_name} Intensity (kg/m²) by {group_by}"
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(group_by, fontsize=12)
    ax.set_ylabel(f"{material_name} Intensity (kg/m²)", fontsize=12)
    
    # Rotate x-axis labels if there are many categories
    if df[group_by].nunique() > 4:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Before saving, check if figure contains data
    print(f"Debug: Axes children count: {len(ax.get_children())}")
    
    # Save the figure
    if output_path is None:
        # Use default output folder from config
        material_name_slug = material_name.lower().replace(" ", "_")
        group_by_slug = group_by.lower().replace(" ", "_")
        filename = f"{material_name_slug}_intensity_by_{group_by_slug}.png"
        output_path = os.path.join(config.RESULTS_OUTPUT_PATH, "plots", filename)
    
    # Add debug prints
    print(f"Debug: Attempting to save figure to {output_path}")
    print(f"Debug: Directory path: {os.path.dirname(output_path)}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Success: Figure saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
    
    return fig

def analyze_any_materials(materials_of_interest: List[str], bom_path=None, group_by='Occupation', output_dir=None):
    """
    Analyze intensity for specified materials in the BOM file and generate plots.
    
    Args:
        materials_of_interest (List[str]): List of material names to analyze. Material names are 
            case-insensitive and should match column names in the BOM file.
        bom_path (str, optional): Path to the BOM file. Defaults to config.BOM_PATH.
        group_by (str, optional): Column to group by for analysis. Defaults to 'Occupation'.
        output_dir (str, optional): Directory to save plots. Defaults to config.RESULTS_OUTPUT_PATH/plots/material_analysis.
    
    Returns:
        dict: Dictionary mapping material names to their figure objects.
    """
    
    # Set default paths
    if bom_path is None:
        bom_path = config.BOM_PATH
    
    if output_dir is None:
        output_dir = os.path.join(config.RESULTS_OUTPUT_PATH, "plots", "material_analysis")
    
    # Add debug prints
    print(f"Debug: BOM path: {bom_path}")
    print(f"Debug: Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load BOM data
    try:
        raw_bom_df = pd.read_csv(bom_path)
        print(f"Debug: Successfully loaded BOM with {len(raw_bom_df)} rows")
    except Exception as e:
        print(f"Error loading BOM file: {str(e)}")
        return {}
    
    # Convert user-provided material names to lowercase for case-insensitive comparison
    materials_lower = [mat.lower() for mat in materials_of_interest]
    
    # Find matching material columns in the BOM file (case-insensitive)
    material_cols = []
    for col in raw_bom_df.columns:
        if col.lower() in materials_lower:
            material_cols.append(col)
    print(f"Debug: Found {len(material_cols)} matching materials")
    # Check if any materials were not found
    found_materials = [col.lower() for col in material_cols]
    not_found = [mat for mat in materials_lower if mat not in found_materials]
    if not_found:
        print(f"Warning: The following materials were not found in the BOM: {', '.join(not_found)}")
    
    if not material_cols:
        print("No matching materials found in the BOM file.")
        return {}
    
    # Dictionary to store figure objects
    figures = {}
    
    # Analyze each material
    for material_col in material_cols:
        material_name = material_col.title()
        output_path = os.path.join(output_dir, f"{material_name.lower().replace(' ', '_')}_by_{group_by.lower().replace(' ', '_')}.png")
        
        try:
            fig = create_material_intensity_boxplots(
                df=raw_bom_df,
                material_col=material_col,
                group_by=group_by,
                title=f"{material_name} Intensity by {group_by}",
                output_path=output_path
            )
            figures[material_name] = fig
            plt.close(fig)  # Close the figure to free memory
            print(f"Generated analysis for {material_name}")
        except Exception as e:
            print(f"Error analyzing {material_name}: {e}")
    
    print(f"Analysis complete. {len(figures)} material plots saved to {output_dir}")
    return figures


if __name__ == "__main__":
    
    # Set matplotlib style for better looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Analyze specific materials by Occupation
    analyze_any_materials(
        materials_of_interest=['steel', 'concrete', 'aluminium'],
        group_by='Occupation',
        output_dir=os.path.join(config.RESULTS_OUTPUT_PATH, "plots", "Occupation_analysis")
    )
    
    # Optionally analyze by other groupings
    analyze_any_materials(
        materials_of_interest=['copper', 'wood'],
        group_by='Country', 
        output_dir=os.path.join(config.RESULTS_OUTPUT_PATH, "plots", "country_analysis")
    )
    
    # Analyze multiple materials by building type
    analyze_any_materials(
        materials_of_interest=['steel', 'concrete', 'wood', 'glass', 'plastics'],
        group_by='building_type',
        output_dir=os.path.join(config.RESULTS_OUTPUT_PATH, "plots", "building_type_analysis")
    )
