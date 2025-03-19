"""
This module contains functions for parsing input files.
"""

"""
================
Import libraries
================
"""
import pandas as pd
import os
from config import config

def extract_bom_header():
    """
    Extracts the header from the BOM file.
    
    Returns:
        list: List of column names from the BOM file.
    """
    try:
        # Read only the first row of the CSV file to get the headers
        bom_df = pd.read_csv(config.BOM_PATH, nrows=0)
        # Convert the column names to a list
        headers = list(bom_df.columns)
        return headers
    except Exception as e:
        print(f"Error extracting BOM header: {e}")
        return []

def parse_rasmi_bom(bom_headers, rasmi_bom_path=None):
    """
    Parses the RASMI BOM file and maps it to the standard BOM format.
    
    Args:
        bom_headers (list): List of column names from the standard BOM file.
        rasmi_bom_path (str, optional): Path to the RASMI BOM file. 
                                        Defaults to config.RASMI_BOM_PATH.
    
    Returns:
        pandas.DataFrame: Parsed and mapped RASMI BOM data.
    """
    if rasmi_bom_path is None:
        rasmi_bom_path = config.RASMI_BOM_PATH
    
    try:
        # Map RASMI data to standard BOM format
        # Read all sheets from the RASMI BOM file
        rasmi_sheets = pd.read_excel(rasmi_bom_path, sheet_name=None)
        
        # Create an empty dataframe with the standard BOM headers
        mapped_df = pd.DataFrame(columns=bom_headers)
        
        # Loop through each sheet in the RASMI file
        for sheet_name, sheet_data in rasmi_sheets.items():
            # Check if the sheet name (lowercase) matches any of the BOM headers (lowercase)
            for header in bom_headers:
                if sheet_name.lower() == header.lower():
                    # If there's a match, map the data from this sheet to the corresponding column
                    if not sheet_data.empty:
                        # Assuming the sheet contains data we want to aggregate
                        # This might need adjustment based on the actual structure of the sheet
                        # mapped_df[header] = sheet_data.sum().iloc[0] if len(sheet_data.sum()) > 0 else 0
                    
                    # Log the mapping for debugging
                    print(f"Mapped sheet '{sheet_name}' to BOM header '{header}'")
        
        # Handle required metadata columns if they exist in the RASMI data
        # This assumes there's a main sheet with building metadata
        if 'building_metadata' in rasmi_sheets:
            metadata = rasmi_sheets['building_metadata']
            for col in metadata.columns:
                if col.lower() in [h.lower() for h in bom_headers]:
                    # Find the exact case-matching header
                    matching_header = next(h for h in bom_headers if h.lower() == col.lower())
                    mapped_df[matching_header] = metadata[col]
        
        return mapped_df
    except Exception as e:
        print(f"Error parsing RASMI BOM file: {e}")
        return pd.DataFrame(columns=bom_headers)
