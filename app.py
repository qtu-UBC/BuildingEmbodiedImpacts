import streamlit as st
import os
import sys
import pandas as pd
from pathlib import Path
from config import config

# Add parent directory to path to import local modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config.config import (
    BOM_PATH,
    IMPACT_FACTORS_PATH,
    BOM_IMPACT_MAPPING_PATH,
    RECIPE_TEMPLATE_PATH,
    RESULTS_OUTPUT_PATH
)

from utilities.helper_funcs import (
    recipe_sheets_gen,
    impact_calc_wrapper_manual,
    store_results,
    merge_sheets,
    process_for_plot
)

from utilities.calculator import calculate_embodied_impacts

st.set_page_config(page_title="Building Embodied Impacts Calculator", layout="wide")

st.title("Building Embodied Impacts Calculator")

st.write("""
This app calculates the embodied environmental impacts of buildings based on their bill of materials.
Upload your input files below to get started.
""")


# File upload widgets
uploaded_bom = st.file_uploader("Upload Buildings BOM (CSV)", type="csv")
uploaded_impact_factors = st.file_uploader("Upload Impact Factors (Excel)", type="xlsx")
uploaded_mapping = st.file_uploader("Upload BOM-Impact Mapping (Excel)", type="xlsx")
uploaded_recipe = st.file_uploader("Upload Recipe Template (Excel)", type="xlsx")

# Process uploaded files if all are present
if all([uploaded_bom, uploaded_impact_factors, uploaded_mapping, uploaded_recipe]):
    try:
        # Read uploaded files
        raw_bom_df = pd.read_csv(uploaded_bom)
        
        # Read impact factors (returns dict of dataframes)
        raw_mat_impacts = pd.read_excel(uploaded_impact_factors, sheet_name=None)
        mat_impact_df = raw_mat_impacts['ei_formatted']
        
        # Read mapping file
        manual_mapping_final_df = pd.read_excel(uploaded_mapping, header=None, index_col=0)
        manual_mapping_final = list(manual_mapping_final_df.to_dict().values())[0]
        
        # Read and format recipe template
        recipe_template_df = pd.read_excel(uploaded_recipe, header=None).T
        recipe_template_df.columns = recipe_template_df.iloc[0]
        recipe_template_df = recipe_template_df[1:]
        recipe_template_df.reset_index(drop=True, inplace=True)

        # Generate recipe sheets
        recipe_sheets =  recipe_sheets_gen(
            save_name='recipe_sheets.xlsx',
            storage_path=config.RESULTS_OUTPUT_PATH,
            archetype_db=raw_bom_df,
            recipe_template=recipe_template_df
        )
        
        results = {}
        for recipe_name, recipe_df in recipe_sheets.items():
            # Calculate impacts using the wrapper function
            impact_results = impact_calc_wrapper_manual(
                recipe_df=recipe_df,
                bom_df=raw_bom_df,
                impact_df=mat_impact_df,
                manual_mapping=manual_mapping_final
            )
            
            # Store results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{recipe_name}_{timestamp}.csv"
            filepath = os.path.join(RESULTS_OUTPUT_PATH, filename)
            
            store_results(
                results_dict=impact_results,
                recipe_df=recipe_df,
                output_path=filepath
            )
            
            results[recipe_name] = impact_results
            
        # Merge all result sheets
        merged_results = merge_sheets(RESULTS_OUTPUT_PATH)
        merged_filepath = os.path.join(RESULTS_OUTPUT_PATH, f"merged_results_{timestamp}.csv")
        merged_results.to_csv(merged_filepath, index=False)
        
        if results:
            st.success("Calculation completed successfully!")
            st.write("Results have been saved to the output directory.")
            
            st.write("Preview of results:")
            st.dataframe(merged_results)
            
            # Add download button for results
            st.download_button(
                label="Download Results",
                data=merged_results.to_csv(index=False),
                file_name=f"building_impacts_results_{timestamp}.csv",
                mime="text/csv"
            )
                
    except Exception as e:
        st.error(f"An error occurred during calculation: {str(e)}")

# Create dropdowns for recipe attributes if files are uploaded
if uploaded_bom and uploaded_recipe:
    try:
        # Read recipe template and BOM file
        recipe_df = pd.read_excel(uploaded_recipe)
        bom_df = pd.read_csv(uploaded_bom)
        
        # Create columns for filters
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            # Get unique occupation values and create dropdown
            occupations = sorted(bom_df['Occupation'].unique().tolist())
            selected_occupation = st.selectbox(
                "Select Occupation",
                options=occupations,
                help="Choose the building occupation type"
            )
            
        with col2:
            # Get unique building type values and create dropdown
            building_types = sorted(bom_df['building_type'].unique().tolist())
            selected_building_type = st.selectbox(
                "Select Building Type",
                options=building_types,
                help="Choose the building type"
            )
            
        with col3:
            # Get unique country values and create dropdown
            countries = sorted(bom_df['Country'].unique().tolist())
            selected_country = st.selectbox(
                "Select Country",
                options=countries,
                help="Choose the country"
            )
            
        with col4:
            # Get unique region values and create dropdown
            regions = sorted(bom_df['Region'].unique().tolist())
            selected_region = st.selectbox(
                "Select Region",
                options=regions,
                help="Choose the region"
            )

        # Add Create BoM button
        if st.button("Create BoM"):
            # Filter BOM based on selected attributes
            filtered_bom = bom_df[
                (bom_df['occupation'] == selected_occupation) &
                (bom_df['building_type'] == selected_building_type) &
                (bom_df['Country'] == selected_country) &
                (bom_df['Region'] == selected_region)
            ]
            
            if filtered_bom.empty:
                st.warning("No matching buildings found for the selected criteria.")
            else:
                st.success(f"Found {len(filtered_bom)} matching buildings")
                
                # If more than one row, calculate mean values
                if len(filtered_bom) > 1:
                    st.info("Multiple buildings found. Showing averaged values.")
                    # Exclude non-numeric columns before calculating mean
                    numeric_cols = filtered_bom.select_dtypes(include=['float64', 'int64']).columns
                    averaged_bom = pd.DataFrame(filtered_bom[numeric_cols].mean()).T
                    
                    # Add back non-numeric columns (using values from first row)
                    non_numeric_cols = filtered_bom.select_dtypes(exclude=['float64', 'int64']).columns
                    for col in non_numeric_cols:
                        averaged_bom[col] = filtered_bom[col].iloc[0]
                    
                    st.dataframe(averaged_bom)
                else:
                    st.dataframe(filtered_bom)
                
    except Exception as e:
        st.error(f"Error processing files: Some numeric columns contain text values. Please ensure all numeric columns contain only numbers.")
        # Add detailed error info
        if hasattr(e, 'args') and len(e.args) > 0:
            st.error(f"Error details: {e.args[0]}")
            if isinstance(e, TypeError):
                # Try to identify the problematic column
                try:
                    error_msg = str(e)
                    if "could not convert string to float" in error_msg:
                        # Extract column name if present in error message
                        import re
                        match = re.search(r"'([^']*)'", error_msg)
                        if match:
                            column_name = match.group(1)
                            st.error(f"Column '{column_name}' contains text values instead of numbers")
                        else:
                            st.error("Please check your data for any text values in numeric columns")
                    else:
                        st.error("Please check your data for any text values in numeric columns")
                except:
                    st.error("Please check your data for any text values in numeric columns")




if st.button("Calculate Impacts"):
    if uploaded_bom and uploaded_impact_factors and uploaded_mapping and uploaded_recipe:
        try:
            # Save uploaded files temporarily
            with open(BOM_PATH, "wb") as f:
                f.write(uploaded_bom.getvalue())
            with open(IMPACT_FACTORS_PATH, "wb") as f:
                f.write(uploaded_impact_factors.getvalue())
            with open(BOM_IMPACT_MAPPING_PATH, "wb") as f:
                f.write(uploaded_mapping.getvalue())
            with open(RECIPE_TEMPLATE_PATH, "wb") as f:
                f.write(uploaded_recipe.getvalue())

            # Calculate impacts
            results = calculate_embodied_impacts()
            
            # Display results
            st.success("Calculation completed successfully!")
            
            # Show results summary
            if os.path.exists(RESULTS_OUTPUT_PATH):
                result_files = os.listdir(RESULTS_OUTPUT_PATH)
                for file in result_files:
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(RESULTS_OUTPUT_PATH, file))
                        st.write(f"Results from {file}:")
                        st.dataframe(df)
                        
                        # Add download button for each result file
                        with open(os.path.join(RESULTS_OUTPUT_PATH, file), 'rb') as f:
                            st.download_button(
                                label=f"Download {file}",
                                data=f,
                                file_name=file,
                                mime='text/csv'
                            )
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
    else:
        st.warning("Please upload all required files before calculating impacts.")
