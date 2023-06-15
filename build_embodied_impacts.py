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
    if not os.path.exists(os.path.sep.join([storage_path,save_xlsx_name])):
        recipe_sheets_gen(save_xlsx_name,storage_path,raw_bom_df,recipe_template_df)

    # generate recipe_sheets list
    recipe_archetype = pd.read_excel(os.path.sep.join([storage_path,save_xlsx_name]), header = None, sheet_name=None)
    recipe_sheets = []
    for sheet_name in recipe_archetype.keys():
        tmp_sheet = recipe_archetype[sheet_name].T
        tmp_sheet.columns = tmp_sheet.iloc[0]
        tmp_sheet = tmp_sheet[1:]
        tmp_sheet.reset_index(drop=True, inplace=True)
        recipe_sheets.append(tmp_sheet)

    # # [Scenario] Baseline
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='baseline_0_0', 
    #     file_name_retained=['WB_BoM','baseline'])

    """ === Recycled Content (RC) === """
    # # [Scenario] 20% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_2', 
    #     file_name_retained=['WB_BoM','RC_0_2'],strategy_dict={'change recycled content': [{'ALL': 0.2}]})

    # # [Scenario] 50% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_5', 
    #     file_name_retained=['WB_BoM','RC_0_5'],strategy_dict={'change recycled content': [{'ALL': 0.5}]})

    # # [Scenario] 80% virgin materials replaced by recycled materials
    # scenario_analysis(raw_bom_df, recipe_sheets, mat_impact_df, manual_mapping_final,strategy_info='RC_0_8', 
    #     file_name_retained=['WB_BoM','RC_0_8'],strategy_dict={'change recycled content': [{'ALL': 0.8}]})

    # [process data for plot]
    # merged sheets of interest [Hard-coded]
    merged_xlsx_of_interest = ['baseline_0_0_2023-06-14_06.xlsx','RC_0_2_2023-06-14_06.xlsx','RC_0_5_2023-06-14_06.xlsx',
    'RC_0_8_2023-06-14_06.xlsx']

    # dict to store the processed df for plot
    df_to_plot_dict = {}

    # loop over lcia of interest
    lcia_of_interest_list = ['ReCiP_no LT_ GWP100','ReCiP_no LT_ WDP','ReCiP_no LT_ FDP'] # can add other impact cateogry of interest
    for lcia_of_interest in lcia_of_interest_list:
        # loop over each xlsx sheet of interest
        for merged_sheet_name in merged_xlsx_of_interest:
            # load xlsx file from drive
            loaded_sheet_dict = pd.read_excel(os.path.sep.join([storage_path,merged_sheet_name]),sheet_name=None)
            # scenario information
            tmp = merged_sheet_name.split("_")
            # quick hack [Hard-coded]
            if tmp[0] == 'baseline':
                tmp[0] = 'RC'
            scenario_info = {tmp[0]: float(".".join([tmp[1],tmp[2]]))}
            # feed the lcia sheet of interest to processing function 
            processed_df = process_for_plot(loaded_sheet_dict[lcia_of_interest],mat_cols,scenario_info) # mat_cols defined at beginning of this script
            # add processed df to the dict
            df_to_plot_dict[merged_sheet_name] = processed_df

        # [save the results to a single file]
        # get the current date and time
        now = datetime.datetime.now()
        # format the date and time as a string
        date_string = now.strftime("%Y-%m-%d_%H")

        file_save_name = "_".join(['RC_scenarios_combined',lcia_of_interest,date_string,'.xlsx'])
        file_save_name = os.path.sep.join([storage_path, file_save_name])

        # create the ExcelWriter obj
        writer = pd.ExcelWriter(file_save_name, engine='xlsxwriter')

        # Loop through the dictionary and save each DataFrame to a separate sheet
        for sheet_name, df in df_to_plot_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)

        # Save the Excel file
        writer.save()

