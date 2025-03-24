"""
This script can be used to calculate impact assessment results for building archetypes

This script fulfills the following purposes
 - run scenario analysis
 - generate plots


"""

"""
================
Import libraries
================
"""
from utilities.helper_funcs import recipe_sheets_gen, impact_calc_wrapper_manual, store_results, merge_sheets,process_for_plot
from config import config
import pandas as pd
import numpy as np
import os
from typing import List, Dict
import datetime


"""
=============
prepare files
=============
"""
# load BoM data
raw_bom_df = pd.read_csv(config.BOM_PATH)

# load impact factors data
raw_mat_impacts = pd.read_excel(config.IMPACT_FACTORS_PATH,sheet_name=None) # returns a dict
mat_impact_df = raw_mat_impacts['ei_formatted']

# load BoM-impact factors mapping data
manual_mapping_final_df = pd.read_excel(config.BOM_IMPACT_MAPPING_PATH,header=None, index_col=0)
manual_mapping_final = list(manual_mapping_final_df.to_dict().values())[0]
manual_mapping_final

# load recipe template
recipe_template_df = pd.read_excel(config.RECIPE_TEMPLATE_PATH, header = None).T
# format the template
recipe_template_df.columns = recipe_template_df.iloc[0]
recipe_template_df = recipe_template_df[1:]
recipe_template_df.reset_index(drop=True, inplace=True)

# set output paths
log_path = config.LOG_OUTPUT_PATH
storage_path = config.RESULTS_OUTPUT_PATH


""" SPECIAL SETUP for Heeren & Fishman 2018 """
# format the material column headers
mat_name_headers = "steel,copper,aluminum,unspecified_metal,wood,paper_cardboard,straw,concrete,cement,aggregates,brick,mortar_plaster,mineral_fill,plaster_board_gypsum,Adobe,Asphalt,Bitumen,natural_stone,cement_asbestos,Clay,siding_unspecified,Ceramics,Glass,Plastics,Polystyrene,PVC,Lineoleum,Carpet,Heraklith,Mineral_wool,insulation_unspecified,other_unspecified_material"
mat_name_headers = [name.strip() for name in mat_name_headers.split(",")]
mat_name_headers_dict = {old_name:f"MAT_{old_name}" for old_name in mat_name_headers} # add suffix to the materials of interest
raw_bom_df.rename(columns=mat_name_headers_dict, inplace=True)
recycled_mat_names = ["".join([mat_col_name,'_recycled']) for mat_col_name in raw_bom_df.columns if mat_col_name.startswith("MAT_")] # create a version of recycled materials [CAUTION] this may produce 'duplicated' col if there is already a recycled material column
for new_col_name in recycled_mat_names:
    raw_bom_df[new_col_name] = 0.0

# data clean up
# remove the whitespace in dataframe
raw_bom_df = raw_bom_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# remove the rows where both 'Occupation' and 'building_type' have missing values -> this removes 22 rows from the original 578 rows
raw_bom_df = raw_bom_df.query('Occupation.notnull() | building_type.notnull()')

# remove the rows where 'Occupation' type is 'residential, non-residential'
#raw_bom_df = raw_bom_df.query("Occupation != 'residential, non-residential'")

# handle the outliers in 'building_type' column
# fill na with value from 'Occupation' column (if there is NA in 'Occupation' column, then the value will still be NA in the 'building_type' column)
raw_bom_df['building_type'].fillna(raw_bom_df['Occupation'],inplace=True)

# simplify the 'Occupation' to: 'residential', 'non-residential'
raw_bom_df['simplified_Occ'] = raw_bom_df['Occupation'].apply(lambda x: x if x=='residential' else 'non-residential')
raw_bom_df.drop(['Occupation'], axis=1, inplace=True)
raw_bom_df.rename(columns={'simplified_Occ':'Occupation'}, inplace=True)

# replace 'unspecified' with np.nan
raw_bom_df.replace('unspecified',np.nan,inplace=True) 

# correction on countries
correct_dict = {
    'US':'USA',
    'Chili': 'Chile',
    'Hong Kong/China': 'China',
    'New Jersey': 'USA',
    'Michigan area': 'USA',
}
raw_bom_df['Country'].replace(correct_dict,inplace=True)

# replace 'aluminum' in column name with 'aluminium'
for col in raw_bom_df.columns:
    if 'aluminum' in col:
        raw_bom_df.rename(columns={col:col.replace('aluminum','aluminium')},inplace=True) 

# material col headers
mat_cols = ['steel', 'copper', 'aluminium',
       'unspecified_metal', 'wood', 'paper_cardboard', 'straw', 'concrete',
       'cement', 'aggregates', 'brick', 'mortar_plaster', 'mineral_fill',
       'plaster_board_gypsum', 'Adobe', 'Asphalt', 'Bitumen', 'natural_stone',
       'cement_asbestos', 'Clay', 'siding_unspecified', 'Ceramics', 'Glass',
       'Plastics', 'Polystyrene', 'PVC', 'Lineoleum', 'Carpet', 'Heraklith',
       'Mineral_wool', 'insulation_unspecified', 'other_unspecified_material',
       'steel_recycled', 'copper_recycled', 'aluminium_recycled',
       'unspecified_metal_recycled', 'wood_recycled',
       'paper_cardboard_recycled', 'straw_recycled', 'concrete_recycled',
       'cement_recycled', 'aggregates_recycled', 'brick_recycled',
       'mortar_plaster_recycled', 'mineral_fill_recycled',
       'plaster_board_gypsum_recycled', 'Adobe_recycled', 'Asphalt_recycled',
       'Bitumen_recycled', 'natural_stone_recycled',
       'cement_asbestos_recycled', 'Clay_recycled',
       'siding_unspecified_recycled', 'Ceramics_recycled', 'Glass_recycled',
       'Plastics_recycled', 'Polystyrene_recycled', 'PVC_recycled',
       'Lineoleum_recycled', 'Carpet_recycled', 'Heraklith_recycled',
       'Mineral_wool_recycled', 'insulation_unspecified_recycled',
       'other_unspecified_material_recycled', ]

save_xlsx_name = 'all_archetypes_included_HF.xlsx'


"""
================
Define functions
================
"""

## define a function for scenario analysis
def scenario_analysis(raw_bom_df: pd.DataFrame, recipe_sheets: Dict, mat_impact_df: pd.DataFrame, mat_name_match_dict: Dict,
    strategy_info: str, file_name_retained: List, **kwargs) -> None:
    """
    [Caution] some hard-coded variables
        - 'MAT_': refers to the label added to the raw BoM df during data preparation
        - 'WB_BoM': refers to whole-building level BoM.
        - unit_convert_dict: a dict contains the conversion factors for certain materials (e.g., kg -> m3 for wood)
    Input params:
        - raw_bom_df: original BoM imported from excel sheet, pd.DataFrame
        - recipe_sheets: contains all the individual recipes, list
        - mat_impact_df: contains material names (by different specification lvl) and corresponding impact factors (e.g., kg CO2-eq/unit), pd.DataFrame
        - mat_name_match_dict: contains the {bom_mat_name: name_in_mat_impact_sheet | location | unit}, dict
        - strategy_info: contains strategy information (e.g., 'baseline_0_0'), str
        - file_name_retained: contains the keywords to filter out the irrelevant files before merging sheets (e.g., ['WB_BoM','baseline']), list
        **kwargs
            - strategy_dict: contains {strategy name: [list of changes]}, e.g., {'change recycled content': [{'ALL': 0.2}]}, dict
    """

    # remove the 'MAT_' from bom names
    clean_dict = {bom_name:bom_name[4:] for bom_name in raw_bom_df.columns if bom_name.startswith("MAT_")}
    raw_bom_df_cleaned = raw_bom_df.rename(columns=clean_dict)

    # prepare a unit conversion factors dict
    unit_convert_dict ={
        'wood': 1/500,
        'concrete': 1/2400
    }

    # for all recipes
    all_recipes_results_dict = {}

    # calculate the impacts                                                                                                                     
    for idx, recipe in enumerate(recipe_sheets):
        tmp_key = "_".join(['recipe',str(idx)])
        if tmp_key in all_recipes_results_dict.keys(): # skip the calculation, if the results are already available from preivous execution
            continue
        else:
            if 'strategy_dict' in kwargs:
                total_impacts_by_mat, archetype_labels_list,impact_category_idx_list = impact_calc_wrapper_manual(recipe,raw_bom_df_cleaned, 
                                                                        mat_impact_df=mat_impact_df,mat_name_match_dict=mat_name_match_dict,
                                                                        unit_convert_dict=unit_convert_dict, strategy_dict=kwargs['strategy_dict'])
            else:
                total_impacts_by_mat, archetype_labels_list,impact_category_idx_list = impact_calc_wrapper_manual(recipe,raw_bom_df_cleaned, 
                                                                        mat_impact_df=mat_impact_df,mat_name_match_dict=mat_name_match_dict,
                                                                        unit_convert_dict=unit_convert_dict)
            all_recipes_results_dict[tmp_key] = [total_impacts_by_mat] # store_results function expects list as the argument, if use list(), it will return a list of column name instead

    # save results to local folder
    # generate the strategy information that becomes part of the file name
    # format: scenario_condition (e.g., MS_0.2 -> material substitution, 20%)
    for recipe_idx, total_impacts_by_mat_list in all_recipes_results_dict.items():
        recipe = recipe_sheets[int(recipe_idx[7:])] # [Hard Coding]
        store_results(f'WB_BoM-{strategy_info}-{recipe_idx}', os.path.sep.join([storage_path,'WB-BoM']), total_impacts_by_mat_list,impact_category_idx_list, recipe)

    # merge sheets
    impact_category_dict = {}

    for lcia_tuple in impact_category_idx_list: # need to run a wrapper first to generate impact_category_idx_list
        impact_category_dict[lcia_tuple[0]] = lcia_tuple[1] # {idx: lcia name}
   
    # get the current date and time
    now = datetime.datetime.now()
    # format the date and time as a string
    date_string = now.strftime("%Y-%m-%d_%H")

    # make sure ONLY appropriate results files are pulled, provide a list of keywords below:
    one_sheet_name = "_".join([strategy_info,date_string])
    merge_sheets(storage_path,os.path.sep.join([storage_path,'WB-BoM']),one_sheet_name,impact_category_dict,file_name_retained=file_name_retained)


"""
=================
Scenario analysis
=================
"""

if __name__ == '__main__':
    # generate all recipes
    # if not os.path.exists(os.path.sep.join([storage_path,save_xlsx_name])):
    #     recipe_sheets_gen(save_xlsx_name,storage_path,raw_bom_df,recipe_template_df)

    # # generate recipe_sheets list
    # recipe_archetype = pd.read_excel(os.path.sep.join([storage_path,save_xlsx_name]), header = None, sheet_name=None,
    #     engine='openpyxl')
    # recipe_sheets = []
    # for sheet_name in recipe_archetype.keys():
    #     tmp_sheet = recipe_archetype[sheet_name].T
    #     tmp_sheet.columns = tmp_sheet.iloc[0]
    #     tmp_sheet = tmp_sheet[1:]
    #     tmp_sheet.reset_index(drop=True, inplace=True)
    #     recipe_sheets.append(tmp_sheet)

    # [Scenario] Baseline
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='baseline_0_0', 
    #     file_name_retained=['WB_BoM','baseline'])

    # # prepare list of change_dicts for each strategy
    # virgin_mat_of_interest = ['steel','concrete','cement','brick'] 
    # recycle_mat_of_interest = ['steel_recycled', 'concrete_recycled', 'cement_recycled', 'brick_recycled']
    # mat_sub_of_interest = ['steel', 'concrete']
    # percent_dict = {
    #     '0_1': 0.1,
    #     '0_2': 0.2,
    #     '0_3': 0.3,
    # }


    # # for recycled content (RC), material efficiency (ME) and material substitution (MS)
    # rc_scenario_dict = {}
    # me_scenario_dict = {}
    # ms_scenario_dict = {}
    # for k,v in percent_dict.items():
    #     scenario_name = "_".join(['rc',k])
    #     rc_scenario_dict[scenario_name] = [{name:v} for name in recycle_mat_of_interest] # e.g., {rc_0_2: [{'steel_recycled':0.2}, ...]}
    #     scenario_name = "_".join(['me',k])
    #     me_scenario_dict[scenario_name] = [{name:v} for name in virgin_mat_of_interest] # e.g., {me_0_2: [{'steel':0.2}, ...]}
    #     scenario_name = "_".join(['ms',k])
    #     # currently only support "WOOD to substitute STEEL & CONCRETE"
    #     ms_scenario_dict[scenario_name] = [{name:v} for name in mat_sub_of_interest] # e.g., {ms_0_2: [{'steel':0.2}, ...]}



    # # # """ === Recycled Content (RC) === """
    # # # [Scenario] 10% virgin materials replaced by recycled materials
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_1', 
    # #     file_name_retained=['WB_BoM','RC_0_1'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_1']})

    # # # [Scenario] 20% virgin materials replaced by recycled materials
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_2', 
    # #     file_name_retained=['WB_BoM','RC_0_2'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_2']})

    # # # [Scenario] 30% virgin materials replaced by recycled materials
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_3', 
    # #     file_name_retained=['WB_BoM','RC_0_3'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_3']})

    # # [7/02/2024] re-popluate recycled content percentage based on different assumptions
    # new_percent_dict = {
    #     '0_40': 0.40,
    #     '0_60': 0.60,
    #     '0_80': 0.80,
    #     '0_100': 1.00,
    # }
    # for k,v in new_percent_dict.items():
    #     scenario_name = "_".join(['rc',k])
    #     rc_scenario_dict[scenario_name] = [{name:v} for name in recycle_mat_of_interest] # e.g., {rc_0_2: [{'steel_recycled':0.2}, ...]}

    # # [Scenario] 40% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_40', 
    #     file_name_retained=['WB_BoM','RC_0_40'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_40']})

    # # [Scenario] 60% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_60', 
    #     file_name_retained=['WB_BoM','RC_0_60'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_60']})

    # # [Scenario] 80% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_80', 
    #     file_name_retained=['WB_BoM','RC_0_80'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_80']})

    # # [Scenario] 100% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_100', 
    #     file_name_retained=['WB_BoM','RC_0_100'],strategy_dict={'change recycled content': rc_scenario_dict['rc_0_100']})


    # # """ === Material Efficiency (ME) === """
    # # # [Scenario] 10% reduction in selected materials
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='ME_0_1', 
    # #     file_name_retained=['WB_BoM','ME_0_1'],strategy_dict={'material efficiency': me_scenario_dict['me_0_1']})

    # # # [Scenario] 20% reduction in selected materials
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='ME_0_2', 
    # #     file_name_retained=['WB_BoM','ME_0_2'],strategy_dict={'material efficiency': me_scenario_dict['me_0_2']})

    # # # [Scenario] 30% reduction in selected materials
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='ME_0_3', 
    # #     file_name_retained=['WB_BoM','ME_0_3'],strategy_dict={'material efficiency': me_scenario_dict['me_0_3']})

    # # [7/02/2024] re-popluate material efficiency percentage based on different assumptions
    # new_percent_dict = {
    #     '0_40': 0.40,
    #     '0_50': 0.50,
    # }
    # for k,v in new_percent_dict.items():
    #     scenario_name = "_".join(['me',k])
    #     me_scenario_dict[scenario_name] = [{name:v} for name in virgin_mat_of_interest] # e.g., {me_0_2: [{'steel':0.2}, ...]}

    # # [Scenario] 40% reduction in selected materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='ME_0_40', 
    #     file_name_retained=['WB_BoM','ME_0_40'],strategy_dict={'material efficiency': me_scenario_dict['me_0_40']})

    # # [Scenario] 50% reduction in selected materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='ME_0_50', 
    #     file_name_retained=['WB_BoM','ME_0_50'],strategy_dict={'material efficiency': me_scenario_dict['me_0_50']})


    # """ === Material Substitution (MS) === """
    # # # [Scenario] 10% of selected materials substituted
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='MS_0_1', 
    # #     file_name_retained=['WB_BoM','MS_0_1'],strategy_dict={'material substitution': ms_scenario_dict['ms_0_1']})

    # # # [Scenario] 20% of selected materials substituted
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='MS_0_2', 
    # #     file_name_retained=['WB_BoM','MS_0_2'],strategy_dict={'material substitution': ms_scenario_dict['ms_0_2']})

    # # # [Scenario] 30% of selected materials substituted
    # # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='MS_0_3', 
    # #     file_name_retained=['WB_BoM','MS_0_3'],strategy_dict={'material substitution': ms_scenario_dict['ms_0_3']})

    # # [7/02/2024] re-popluate material substitution percentage based on different assumptions
    # new_percent_dict = {
    #     '0_40': 0.40,
    #     '0_60': 0.60,
    # }
    # for k,v in new_percent_dict.items():
    #     scenario_name = "_".join(['ms',k])
    #     # currently only support "WOOD to substitute STEEL & CONCRETE"
    #     ms_scenario_dict[scenario_name] = [{name:v} for name in mat_sub_of_interest] # e.g., {ms_0_25: [{'steel':0.25}, ...]}

    # # [Scenario] 40% of selected materials substituted
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='MS_0_40', 
    #     file_name_retained=['WB_BoM','MS_0_40'],strategy_dict={'material substitution': ms_scenario_dict['ms_0_40']})

    # # [Scenario] 60% of selected materials substituted
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='MS_0_60', 
    #     file_name_retained=['WB_BoM','MS_0_60'],strategy_dict={'material substitution': ms_scenario_dict['ms_0_60']})


    # [process data for plot]
    # merged sheets of interest [Hard-coded]
    # merged_xlsx_of_interest = ['ME_0_40_2024-07-02_20.xlsx','ME_0_50_2024-07-02_20.xlsx','MS_0_40_2024-07-02_20.xlsx','MS_0_60_2024-07-02_20.xlsx',
    #     'RC_0_40_2024-07-02_20.xlsx','RC_0_60_2024-07-02_20.xlsx','RC_0_80_2024-07-02_20.xlsx','RC_0_100_2024-07-02_20.xlsx',]

    # # dict to store the processed df for plot
    # df_to_plot_dict = {}

    # # loop over lcia of interest
    # lcia_of_interest_list = ['ReCiP_no LT_ GWP100','ReCiP_no LT_ WDP','ReCiP_no LT_ FDP', 'TRACI_TRACI_ ecotoxicity'] # can add other impact cateogry of interest
    # for lcia_of_interest in lcia_of_interest_list:
    #     # loop over each xlsx sheet of interest
    #     for merged_sheet_name in merged_xlsx_of_interest:
    #         # load xlsx file from drive
    #         loaded_sheet_dict = pd.read_excel(os.path.sep.join([storage_path,merged_sheet_name]),sheet_name=None)
    #         # scenario information
    #         tmp = merged_sheet_name.split("_")
    #         # quick hack [Hard-coded]
    #         if tmp[0] == 'baseline':
    #             tmp[0] = 'RC'
    #         scenario_info = {tmp[0]: float(".".join([tmp[1],tmp[2]]))}
    #         # feed the lcia sheet of interest to processing function 
    #         processed_df = process_for_plot(loaded_sheet_dict[lcia_of_interest],mat_cols,scenario_info) # mat_cols defined at beginning of this script
    #         # add processed df to the dict
    #         df_to_plot_dict[merged_sheet_name] = processed_df

    #     # [save the results to a single file]
    #     # get the current date and time
    #     now = datetime.datetime.now()
    #     # format the date and time as a string
    #     date_string = now.strftime("%Y-%m-%d_%H")

    #     file_save_name = "_".join(['ALL_scenarios_combined',lcia_of_interest,date_string,'.xlsx'])
    #     file_save_name = os.path.sep.join([storage_path, file_save_name])

    #     # create the ExcelWriter obj
    #     writer = pd.ExcelWriter(file_save_name, engine='xlsxwriter')

    #     # Loop through the dictionary and save each DataFrame to a separate sheet
    #     for sheet_name, df in df_to_plot_dict.items():
    #         df.to_excel(writer, sheet_name=sheet_name)

    #     # Save the Excel file
    #     writer.close()

        # Analyze material intensity across building types
        print("Analyzing material intensity across building types...")
        
        # Define materials of interest for analysis
        materials_to_analyze = ['steel', 'concrete', 'wood', 'copper', 'aluminum']
        
        # Convert to MAT_ format to match the column names in the BOM dataframe
        mat_columns = [f"MAT_{material}" for material in materials_to_analyze]
        
        # Make sure the output directory exists
        plot_output_dir = os.path.join(storage_path, "plots", "material_intensity")
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Use the raw_bom_df for material intensity analysis
        # This dataframe should already have the necessary columns for building types and occupancy
        print(f"Analyzing material intensity for {len(materials_to_analyze)} materials...")
        
        # Create boxplots for each material by Occupation
        for material, mat_col in zip(materials_to_analyze, mat_columns):
            if mat_col in raw_bom_df.columns:
                try:
                    print(f"Creating boxplot for {material} by Occupation...")
                    fig = create_material_intensity_boxplots(
                        df=raw_bom_df,
                        material_col=mat_col,
                        group_by="Occupation",
                        output_path=os.path.join(plot_output_dir, f"{material}_by_occupation.png"),
                        title=f"{material.title()} Intensity by Building Occupation"
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"Error creating boxplot for {material} by Occupation: {str(e)}")
            else:
                print(f"Warning: Material column '{mat_col}' not found in BOM dataframe")
        
        # Create boxplots for each material by Building Type
        if "Building Type" in raw_bom_df.columns:
            for material, mat_col in zip(materials_to_analyze, mat_columns):
                if mat_col in raw_bom_df.columns:
                    try:
                        print(f"Creating boxplot for {material} by Building Type...")
                        fig = create_material_intensity_boxplots(
                            df=raw_bom_df,
                            material_col=mat_col,
                            group_by="Building Type",
                            output_path=os.path.join(plot_output_dir, f"{material}_by_building_type.png"),
                            title=f"{material.title()} Intensity by Building Type"
                        )
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error creating boxplot for {material} by Building Type: {str(e)}")
        else:
            print("Warning: 'Building Type' column not found in BOM dataframe")
        
        # Use the analyze_any_materials function for a more comprehensive analysis
        print("Performing comprehensive material analysis...")
        try:
            from utilities.analyze import analyze_any_materials
            
            # Create a dedicated output directory for the comprehensive analysis
            comprehensive_output_dir = os.path.join(storage_path, "plots", "comprehensive_material_analysis")
            
            # Analyze by Occupation
            analyze_any_materials(
                materials_of_interest=materials_to_analyze,
                bom_path=config.BOM_PATH,
                group_by="Occupation",
                output_dir=comprehensive_output_dir
            )
            
            # Analyze by Building Type if the column exists
            if "Building Type" in raw_bom_df.columns:
                analyze_any_materials(
                    materials_of_interest=materials_to_analyze,
                    bom_path=config.BOM_PATH,
                    group_by="Building Type",
                    output_dir=comprehensive_output_dir
                )
        except Exception as e:
            print(f"Error in comprehensive material analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("Material intensity analysis completed.")
        # Analyze scenario results with boxplots
