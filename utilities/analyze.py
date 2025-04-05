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
                                 Can be "Occupation", "Country", or a list ["Occupation", "Country"]
                                 for grouped analysis.
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
    
    # Handle different group_by options
    if isinstance(group_by, list):
        # Multiple grouping columns
        for col in group_by:
            if col not in df.columns:
                raise ValueError(f"Group column '{col}' not found in DataFrame")
        # Create a combined grouping column
        df['_combined_group'] = df[group_by].apply(lambda x: ' - '.join(x.astype(str)), axis=1)
        actual_group_by = '_combined_group'
    else:
        # Single grouping column
        if group_by not in df.columns:
            raise ValueError(f"Group column '{group_by}' not found in DataFrame")
        actual_group_by = group_by
    
    # Remove rows with NaN or infinite values
    df = df[np.isfinite(df[material_col])]
    
    # Identify extreme outliers (values beyond 3 IQRs from the quartiles)
    Q1 = df[material_col].quantile(0.25)
    Q3 = df[material_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Identify and print extreme outliers
    outliers = df[(df[material_col] < lower_bound) | (df[material_col] > upper_bound)]
    if not outliers.empty:
        print(f"\nExtreme outliers detected for {material_col}:")
        print(f"Values outside range: {lower_bound:.2f} to {upper_bound:.2f}")
        print(outliers[[actual_group_by, material_col]].to_string())
        print(f"Total outliers: {len(outliers)} out of {len(df)} data points")
    
    # Remove extreme outliers for visualization
    df_filtered = df[(df[material_col] >= lower_bound) & (df[material_col] <= upper_bound)]
    
    # Add more detailed debug prints
    print(f"Debug: DataFrame shape after removing outliers: {df_filtered.shape}")
    print(f"Debug: Number of finite intensity values: {np.sum(np.isfinite(df_filtered[material_col]))}")
    print(f"Debug: Unique {actual_group_by} values: {df_filtered[actual_group_by].unique()}")
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    print("Debug: Figure created")
    
    # Create the boxplot
    sns.boxplot(
        x=actual_group_by,
        y=material_col,
        data=df_filtered,
        ax=ax,
        palette=palette
    )
    
    # Add individual data points for better visualization
    sns.stripplot(
        x=actual_group_by,
        y=material_col,
        data=df_filtered,
        ax=ax,
        color='black',
        alpha=0.5,
        size=4,
        jitter=True
    )
    
    # Set title and labels
    material_name = material_col.replace("MAT_", "").title()
    if title is None:
        if isinstance(group_by, list):
            title = f"{material_name} Intensity (kg/m²) by {' and '.join(group_by)}"
        else:
            title = f"{material_name} Intensity (kg/m²) by {group_by}"
    
    ax.set_title(title, fontsize=14)
    if isinstance(group_by, list):
        ax.set_xlabel(' and '.join(group_by), fontsize=12)
    else:
        ax.set_xlabel(group_by, fontsize=12)
    ax.set_ylabel(f"{material_name} Intensity (kg/m²)", fontsize=12)
    
    # Rotate x-axis labels if there are many categories
    if df_filtered[actual_group_by].nunique() > 4:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Before saving, check if figure contains data
    print(f"Debug: Axes children count: {len(ax.get_children())}")
    
    # Save the figure
    if output_path is None:
        # Use default output folder from config
        material_name_slug = material_name.lower().replace(" ", "_")
        if isinstance(group_by, list):
            group_by_slug = '_and_'.join([g.lower().replace(" ", "_") for g in group_by])
        else:
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
        group_by (str or list, optional): Column(s) to group by for analysis. Can be a string like 'Occupation'
            or a list of columns like ['Occupation', 'Country']. Defaults to 'Occupation'.
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
    
    # Prepare group_by string for filename
    if isinstance(group_by, list):
        group_by_filename = '_and_'.join([g.lower().replace(' ', '_') for g in group_by])
    else:
        group_by_filename = group_by.lower().replace(' ', '_')
    
    # Analyze each material
    for material_col in material_cols:
        material_name = material_col.title()
        output_path = os.path.join(output_dir, f"{material_name.lower().replace(' ', '_')}_by_{group_by_filename}.png")
        
        try:
            fig = create_material_intensity_boxplots(
                df=raw_bom_df,
                material_col=material_col,
                group_by=group_by,
                title=f"{material_name} Intensity by {' and '.join(group_by) if isinstance(group_by, list) else group_by}",
                output_path=output_path
            )
            figures[material_name] = fig
            plt.close(fig)  # Close the figure to free memory
            print(f"Generated analysis for {material_name}")
        except Exception as e:
            print(f"Error analyzing {material_name}: {e}")
    
    print(f"Analysis complete. {len(figures)} material plots saved to {output_dir}")
    return figures

def show_top_k_boxplots(df, col_to_plot, criteria, k=3, group_by=['Country', 'Occupation']):
    """
    Show the top k results for a given column and group by a list of columns.
    Only includes groups with at least 5 data points.
    """
    # First, print the available columns to debug
    print(f"Debug: Available columns in DataFrame: {df.columns.tolist()}")
    
    # Verify all group_by columns exist
    missing_cols = [col for col in group_by if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing: {missing_cols}")
        # Try to find similar column names (case-insensitive)
        for missing_col in missing_cols:
            similar_cols = [col for col in df.columns if col.lower() == missing_col.lower()]
            if similar_cols:
                print(f"Found similar column name: '{similar_cols[0]}' for '{missing_col}'")
                # Update group_by with the correct column name
                group_by[group_by.index(missing_col)] = similar_cols[0]
            else:
                raise ValueError(f"Required column '{missing_col}' not found in DataFrame")
    
    # Calculate statistics for each group and count the number of data points
    grouped = df.groupby(group_by)
    
    # Get the count of data points in each group
    group_counts = grouped[col_to_plot].count().reset_index()
    group_counts.rename(columns={col_to_plot: 'count'}, inplace=True)
    
    # Calculate the requested statistic for each group
    if criteria == "standard deviation":
        group_stats = grouped[col_to_plot].std().reset_index()
    else:  # default to mean
        group_stats = grouped[col_to_plot].mean().reset_index()
    
    # Merge the statistics with the counts
    group_stats = pd.merge(group_stats, group_counts, on=group_by)
    
    # Filter to only include groups with at least 5 data points
    group_stats = group_stats[group_stats['count'] >= 5]
    
    # Check if we have any groups left after filtering
    if len(group_stats) == 0:
        print(f"Warning: No groups have at least 5 data points for {col_to_plot}")
        return None
    
    # Sort and get top k groups
    group_stats = group_stats.sort_values(by=col_to_plot, ascending=False).head(k)
    
    # Create a filter for the original dataframe
    filter_conditions = None
    for _, row in group_stats.iterrows():
        condition = pd.Series(True, index=df.index)
        for col in group_by:
            condition &= (df[col] == row[col])
        if filter_conditions is None:
            filter_conditions = condition
        else:
            filter_conditions |= condition
    
    # Filter the dataframe
    df_filtered = df[filter_conditions].copy()
    
    # Create the combined group labels
    df_filtered['_combined_group'] = ''
    for idx, row in df_filtered.iterrows():
        labels = [str(row[col]) for col in group_by]
        df_filtered.at[idx, '_combined_group'] = ' - '.join(labels)
    
    # Create the ordered group labels for plotting
    top_k_combined_groups = []
    for _, row in group_stats.iterrows():
        labels = [str(row[col]) for col in group_by]
        top_k_combined_groups.append(' - '.join(labels))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(
        x='_combined_group',
        y=col_to_plot,
        data=df_filtered,
        ax=ax,
        order=top_k_combined_groups
    )
    
    # Add title and labels
    group_by_display = ' & '.join(group_by)
    ax.set_title(f"Top {k} {group_by_display} by {col_to_plot} ({criteria}) - Min 5 data points")
    ax.set_xlabel(group_by_display)
    ax.set_ylabel(col_to_plot)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    group_by_filename = '_'.join(group_by)
    output_path = os.path.join(config.RESULTS_OUTPUT_PATH, "plots", "top_k_analysis", 
                              f"{col_to_plot}_{group_by_filename}_top_{k}_{criteria.lower().replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    
    # Print information about the top k groups
    print(f"\nTop {k} {group_by_display} by {col_to_plot} ({criteria}) with at least 5 data points:")
    for _, row in group_stats.iterrows():
        group_identifier = ', '.join([f"{col}={row[col]}" for col in group_by])
        print(f"{group_identifier}: {row[col_to_plot]:.2f} (n={row['count']})")
    
    return fig

################################################################################

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


