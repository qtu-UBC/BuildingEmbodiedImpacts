import pandas as pd
import os
import matplotlib.pyplot as plt
from config import config
import seaborn as sns

print("Analyzing scenario results with boxplots...")
try:
    from utilities.analyze import show_top_k_boxplots
    
    # Define storage path - add this line
    storage_path = config.RESULTS_OUTPUT_PATH

    # Create output directory for scenario analysis plots
    scenario_output_dir = os.path.join(storage_path, "plots", "scenario_analysis")
    os.makedirs(scenario_output_dir, exist_ok=True)
    
    # Load the scenario results file
    scenario_file_path = os.path.join(storage_path, "ALL_scenarios_combined_ReCiP_no LT_ GWP100_2024-07-02_21_.xlsx")
    print(f"Loading scenario file: {scenario_file_path}")
    
    # Load all sheets from the Excel file
    excel_file = pd.ExcelFile(scenario_file_path)
    sheet_names = excel_file.sheet_names
    
    # Process each sheet
    for sheet_name in sheet_names:
        print(f"Processing sheet: {sheet_name}")
        
        # Read the sheet into a DataFrame
        df = pd.read_excel(scenario_file_path, sheet_name=sheet_name)
        
        # Check if required columns exist
        if "TOTAL impacts" not in df.columns:
            print(f"Warning: 'TOTAL impacts' column not found in sheet {sheet_name}")
            continue
        
        # Check if both required columns exist
        if "Country" in df.columns and "Occupation" in df.columns:
            try:
                print(f"Creating boxplot for {sheet_name} showing TOTAL impacts by country (top 3 per occupation)...")
                
                # Get unique occupations
                occupations = df["Occupation"].unique()
                
                # Create a figure with subplots for each occupation
                fig, axes = plt.subplots(nrows=len(occupations), figsize=(12, 6*len(occupations)))
                if len(occupations) == 1:
                    axes = [axes]  # Convert to list if only one occupation
                
                # Process each occupation
                for i, occupation in enumerate(occupations):
                    # Filter data for this occupation
                    occupation_df = df[df["Occupation"] == occupation]
                    
                    # Calculate top 3 countries by standard deviation for this occupation
                    country_stats = occupation_df.groupby("Country")["TOTAL impacts"].std().reset_index()
                    country_stats = country_stats.sort_values(by="TOTAL impacts", ascending=False)
                    
                    # Only include countries with at least 5 data points
                    country_counts = occupation_df.groupby("Country").size().reset_index(name="count")
                    valid_countries = country_counts[country_counts["count"] >= 5]["Country"].tolist()
                    
                    # Filter to valid countries with enough data points
                    country_stats = country_stats[country_stats["Country"].isin(valid_countries)]
                    
                    # Get top 3 countries
                    top_3_countries = country_stats.head(3)["Country"].tolist()
                    
                    if not top_3_countries:
                        axes[i].text(0.5, 0.5, f"No countries with sufficient data for {occupation}", 
                                    ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f"Occupation: {occupation}")
                        continue
                    
                    # Filter data for top 3 countries in this occupation
                    plot_df = occupation_df[occupation_df["Country"].isin(top_3_countries)]
                    
                    # Create boxplot
                    sns.boxplot(x="Country", y="TOTAL impacts", data=plot_df, ax=axes[i])
                    axes[i].set_title(f"Occupation: {occupation} - Top 3 Countries by Impact Variability")
                    axes[i].set_xlabel("Country")
                    axes[i].set_ylabel("TOTAL impacts")
                    
                    # Add sample size to each country label
                    for j, country in enumerate(top_3_countries):
                        count = len(plot_df[plot_df["Country"] == country])
                        axes[i].get_xticklabels()[j].set_text(f"{country} (n={count})")
                
                plt.tight_layout()
                
                # Save with a descriptive filename
                output_path = os.path.join(scenario_output_dir, f"{sheet_name}_top3_countries_per_occupation.png")
                plt.savefig(output_path, dpi=600)
                plt.close(fig)
                
                print(f"  - Created boxplot for top 3 countries per occupation in {sheet_name}")
                
            except Exception as e:
                print(f"Error creating boxplot for {sheet_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            missing_cols = []
            if "Country" not in df.columns:
                missing_cols.append("Country")
            if "Occupation" not in df.columns:
                missing_cols.append("Occupation")
            print(f"Warning: {', '.join(missing_cols)} column(s) not found in sheet {sheet_name}")
    
    print("Scenario analysis completed.")
except Exception as e:
    print(f"Error in scenario analysis: {str(e)}")
    import traceback
    traceback.print_exc()


