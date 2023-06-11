"""
This script contains the following helper functions:
 - create recipe sheets (i.e., location, building type, etc.): fed to archetype creation function
 - create an archetype with BoM
 - apply strategies (e.g., material substitution)
 - calculate results for multiple impact categories
 - save results to local folder
 - merge result sheets
 - process results for generating plots
 - wrapper for the calculation pipeline: archetype creation, apply strategies, calculate impact category results

Current support the following BoM format:
 - Heeren & Fishman (2018): A database seed for a community-driven material intensity research platform
"""

"""
================
Import libraries
================
"""
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from copy import deepcopy
import itertools
import io
import os
import sys
import datetime
import logging
from config import config


"""
=================
set up the logger
=================
"""
# gets or creates a logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
log_output_path = os.path.sep.join([config.LOG_OUTPUT_PATH,'helper_funcs.log'])
file_handler = logging.FileHandler(log_output_path,mode='w')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)  


"""
================
define functions
================
"""

## define a function for create recipe sheets
def recipe_sheets_gen(save_name: str, storage_path: str, archetype_db: pd.DataFrame, recipe_template: pd.DataFrame) -> None:
	"""
	This function generates recipe sheets (one for each recipe) in a .xlsx file that can later be reformated and used for bom_archetype_gen()

	Input params:
	- save_name: name of the excel file to save, str
	- storage_path: path to the destination folder for storage, str
	- archetype_db: contains the information of all archetypes in a given datasheet (e.g., Heeren & Fishman, 2018), DataFrame
	- recipe_template: contains the characterisitics of an archtype (e.g., Occupation, Region) that are applicable to all archetypes in a given archetype datasheet, DataFrame

	"""

	# combine storage path with file name
	storage_path = os.path.sep.join([storage_path,save_name])

	# create the ExcelWriter obj
	writer = pd.ExcelWriter(storage_path, engine='xlsxwriter')

	# transpose the recipe template, as its original format is: 1st column: characterisitics (e.g., Occupation, Region), 2nd column: values of those characterisitics (e.g., Residential)
	#recipe_template = recipe_template.T -> assume the template has been transposed

	# find the common headers between the two dfs
	commmon_headers = list(set(archetype_db.columns)&set(recipe_template))

	# slice the archetype_db based on the common headers and retain the unique rows
	to_recipe_sheets = archetype_db[commmon_headers].copy()
	to_recipe_sheets = to_recipe_sheets.drop_duplicates()

	# iterate over the slice to create a sheet for each row: Thanks ChatGPT!
	for row in to_recipe_sheets.itertuples():
		sheet_name = "_".join(['recipe_',str(row.Index)])
		sheet = pd.DataFrame([row[1:]], columns=row._fields[1:]).T
		sheet.to_excel(writer, sheet_name=sheet_name, header=None) 
	# save all sheets to the excel file
	writer.save()


## define a function for creating an archetype with BoM
def archetype_gen(bom_df_processed: pd.DataFrame, recipe: pd.DataFrame, select_existing = True) -> (pd.DataFrame, object,list):
	"""
	Input param:
	- bom_df_processed: a dataframe contains the archetype classification and corresponding BoM (preprocessed), pd.DataFrame
	  *currently support:
		- Heeren & Fishman 2018
	- recipe: a dataframe converted from the input excel file that contains the following archetype information:
	  *column headers containing the attributes of interest: building_type, location, assembly_components (multiple col: Foundations, Slab on Grade...), WB_only
		- if 'WD_only' is True, then values of assembly_components will be ignored when generating the archetype
	  *1st row contains the values corresponds to attributes column headers: e.g., office, Canada, M1/M2/M3 for each row of assembly_components
	- select_existing: whether or not select an archetype from existing dataset, boolean
	  *[CAUTION] currently only support 'True'
	Returns:
	- archetype_BoM_df: a dataframe contains BoM information for the archetype of interest, pd.DataFrame
	  *[CAUTION] the df may contain multiple rows
	- archetype_idx: index corresponding to the index of archetypes selected from the bom_df_processed, object
	- archetype_labels: a list of archetype labels that are later used to label the results in visualization, list
	"""

	# get the intersection between headers of the recipe and those of BoM dataframe -> this serves as the available handle for filtering
	common_handles = list(set(recipe.columns)&set(bom_df_processed.columns))
	# log the warning about the columns of recipe that are not included for the query -> this will inform the mismatching between archetype datasheet and recipe template
	logger.info(" =================================================================== ")
	logger.info(f"\n The following elements in the RECIPE template are not included in the archetype dataset: {[el for el in recipe.columns if el not in common_handles]} \n")
	logger.info(" =================================================================== ")

	# parse recipe file (only common headers)
	recipe_dict = recipe[common_handles].to_dict('records')[0] #https://stackoverflow.com/questions/52547805/how-to-convert-dataframe-to-dictionary-in-pandas-without-index
	# create a query entry: [CAUTION] need to account for upper/lower case entries later [CAUTION] also need to be careful when expr contains 'nan' [CAUTION] need to remove the existing quotation marks
	query_ent = []
	for col, val in recipe_dict.items():
		val = str(val) # convert nan (float) to nan (string), so that you can use the count() method [caution] this may inadvertently convert float type (intended) to string
		if val.count('"') == 0 and val.count("'") == 0: # if the string does not contain "" or '' already
			query_ent.append(" == ".join([col,f'"{val}"']))
		elif val.count('"') > 0: # if there are no double quotation marks (can be more than one pairs or odd number of quotation marks)
			query_ent.append(" == ".join([col,f"'{val}'"])) # swtich the quotatioin mark types to avoid error with df.query()
		elif val.count("'") > 0:
			query_ent.append(" == ".join([col,f'"{val}"']))
	query_ent = [f'({x})' for x in query_ent]
	query_ent = " and ".join(query_ent)

	# select from existing archetypes
	if select_existing:
		# replace np.nan in the bom_df_processed with a str 'nan' for matching
		tmp_copy = bom_df_processed.copy()
		tmp_copy = tmp_copy.replace(np.nan,'nan')
		# matching the attributes from the recipe file with those in the BoM dataset 
		selected = pd.DataFrame(tmp_copy.query(query_ent))

		archetype_BoM_df = selected
		archetype_idx = archetype_BoM_df.index
		# prepare an achetype label list for visualization later
		archetype_labels = []
		archetype_build_type,archetype_loc = recipe['building_type'].values[0],recipe['Region'].values[0]
		for idx in archetype_idx:
			archetype_labels.append("_".join([str(archetype_build_type),str(archetype_loc),str(idx)]))

		return archetype_BoM_df, archetype_idx, archetype_labels


## define a function to apply strategies
def apply_strategies(archetype_BoM_df: pd.DataFrame, strategy_dict: dict) -> dict:
	"""
	This function applies strategies to update the material and/or quantity of the original BoM dataframe for a SINGLE recipe
	Input params:
	- archetype_BoM_df: a dataframe contains original BoM inforrmation for the archetype of interest, pd.DataFrame
		*[CAUTION] the df may contain multiple rows
	- strategy_dict: a dict contains {'strategy_name': list of dict_strategy_implementation}
		* for WB-BoM: e.g., {'material efficiency': 
			[{'MAT_straw': quant_change},{'MAT_steel': quant_change}],
				'recycled content change':
			[{'MAT_wood': quant_change},{'MAT_wood_recycled': quant_change}],
			}
		* for Assembly-lvl BoM: e.g., {'material efficiency': 
			[{(index_name,'MAT_straw'): quant_change},{(index_name,'MAT_steel'): quant_change}],
				'recycled content change':
			[{(index_name,'MAT_wood'): quant_change},{(index_name,'MAT_wood_recycled'): quant_change}],
			}
		*[CAUTION]
		- 'quant_change' by convention (of this code pipeline) is a positive fraction (e.g., 0.2)
		- the REDUCTION in quantity is made by multiplying the original quantity with (1-quant_change)
		- currently no constraint of the range of quant_change is implemented -> e.g., if quant_change is > 1, 
		 then you will end up with negative value of a material (i.e., m = m * (1-quant_change))
	Returns:
	- archetype_BoM_df_updated: a dataframe contains the updated BoM of an archetype, pd.DataFrame

	"""
	# initiate the output
	archetype_BoM_df_updated = archetype_BoM_df.copy()
	archetype_BoM_df_updated = archetype_BoM_df_updated.apply(lambda x: pd.to_numeric(x, errors='coerce'))
	archetype_BoM_df_updated = archetype_BoM_df_updated.fillna(0) # this is to prevent the case where recycled material quantity is NaN (NaN + float -> NaN)
	# print(f" this is the intialized version: {archetype_BoM_df_updated['steel']} \n {archetype_BoM_df_updated['steel_recycled']}")

	# currently supported strategies
	current_spt_strategies = ['material efficiency', 'material substitution', 'change recycled content']

	# iterate over strategies:
	for strategy_name in strategy_dict.keys():
		if strategy_name in current_spt_strategies:
			for change_dict in strategy_dict[strategy_name]: # strategy_dict[strategy_name] points to the list of change_dicts to be implemented
				# check if apply strategy to WB-BoM or Assembly-lvl BoM (key is a tuple)
				change_key = list(change_dict.keys())[0]
				change_val = list(change_dict.values())[0]
				if isinstance(change_key,tuple):
					# check index dtype (should be 'object', e.g., 'A1010 Standard Foundations')
					assert archetype_BoM_df_updated.index.dtype == 'object', "[Error] please make sure the BoM df format is for Assembly-lvl BoM"
					# find the cell in the df to make the change
					archetype_BoM_df_updated.loc[change_key[0],change_key[1]] += change_val
				else:
					# check index dtype (should be int64)
					assert archetype_BoM_df_updated.index.dtype == 'int64', "[Error] please make sure the BoM df format is for WB-BoM"
					# [CAUTION] the changes will be applied to ALL achetypes under one recipe
					if change_key == 'ALL':
						# get the col names
						tmp_col_names = archetype_BoM_df_updated.columns
						# apply the same change to all materials
						if strategy_name == 'change recycled content':
							# create a virgin:recycled material dict
							virgin_recycled_dict = {col[:-9]: col for col in tmp_col_names if col.endswith('recycled')}
							# get the change quantity
							tmp_delta = archetype_BoM_df_updated[list(virgin_recycled_dict.keys())] * change_val # e.g., 20% of virgin materials are replaced by recycled materials -> change_val == 0.2
							# add the change quantity to recycled material columns
							for virgin,recycled in virgin_recycled_dict.items():
								archetype_BoM_df_updated[recycled] = archetype_BoM_df_updated[recycled] + tmp_delta[virgin] # '+' is not supported by pd.Series
								archetype_BoM_df_updated[virgin] = archetype_BoM_df_updated[virgin] - tmp_delta[virgin]
						elif strategy_name == 'material efficiency':
							# reduce the material intensity by x percent
							archetype_BoM_df_updated = archetype_BoM_df_updated * (1-change_val)
						else:
							# capture the other conditions
							# log the warning
							logger.info(" =================================================================== ")
							logger.info(f"[Warning]the combination of {change_key} & {strategy_name} is currently not supported \n")
							logger.info(" =================================================================== ")
					else:
						# update the value of each with 'quant_change'
						archetype_BoM_df_updated[change_key] = archetype_BoM_df_updated[change_key].apply(lambda x: x*(1-change_val))
		else:
			# log the warning
			logger.info(" =================================================================== ")
			logger.info(f"[Warning] The strategy {strategy_name} you input is currently NOT supported by the code \n")
			logger.info(" =================================================================== ")
	
	return archetype_BoM_df_updated


## define a function to calculate results for multiple impact categories
def calc_impact_multi(mat_name_match_dict: dict, mat_impact_df: pd.DataFrame, bom_quantity_df: pd.DataFrame, **kwargs) -> (pd.DataFrame,list):
	"""
	Input params:
	- mat_name_match_dict: a dict contains the {bom_mat_name: name_in_mat_impact_sheet | location | unit}, dict
	- mat_impact_df: a dataframe contains material names (by different specification lvl) and corresponding impact factors (e.g., kg CO2-eq/unit), pd.DataFrame
		* unit converstion will be done (i.e., all impacts are XX/kg material) in this function
	- bom_quantity_df: a dataframe contains the columns of 'bom_name' and corresponding 'bom_quantity' of the archetype(s) of interest, pd.DataFrame
	- kwargs:
		* unit_convert_dict: contains the converstion factors for units of bom (e.g., from kg to m3 for concrete), {bom_name: unit_convert_factor}, dict
	- these factors should be used as mutipliers
		* [depreciated] matching_archetype_idx: a list of index corresponding to the index of archetypes in the input bom file
	Returns:
	- total_impacts_by_mat: a dataframe contains the total impact from each material (col) of interest for archetypes (row)
	- impact_category_idx_list: a list of (idx, impact_category_name) of the corrseponding impact category results in the df to plot

	[Cautions]:
	- hard coding for column names: 
		* 'M1', 'M2', 'M3'
	- hard coding for positions: 
		* normalized impact category results starts on index 6
	_ hard coding for string patterns
		* use "_" as the connector for BoM names, e.g., "steel_recycled"
		* use "|" as the connector for mat impact names, e.g, "xxx | GLO | unit"
	"""

	# make a slice of bom_quantity_df to only keep the columns of bom names
	slice_bom_quantity_df = bom_quantity_df[list(mat_name_match_dict.keys())].copy()

	# perform unit conversion, if needed
	if 'unit_convert_dict' in kwargs:
		tmp_bom_cols = [col.lower() for col in slice_bom_quantity_df.columns]
		for k,v in kwargs['unit_convert_dict'].items():
			if k.lower() in tmp_bom_cols:
				slice_bom_quantity_df[k] = slice_bom_quantity_df[k].astype(float).mul(v)
			else:
				# log the warning
				logger.info(" =================================================================== ")
				logger.info(f"[CAUTION] {k} is not in the columns of the bom name df!! \n")
				logger.info(" =================================================================== ")

	# a trick to get rid of empty columns
	col_to_keep = [col_name for col_name in mat_impact_df.columns[6:] if not (mat_impact_df[col_name] == 0).all()] # [hard code warning] normalized impact category results starts on index 6
	impact_category_idx_list = [ (idx, header) for idx, header in enumerate(col_to_keep)]
	mat_impact_df = pd.concat([mat_impact_df.iloc[:,:6], mat_impact_df[col_to_keep]], axis=1)
	# remove duplicated rows
	mat_impact_df.drop_duplicates(inplace=True)

	# collapse the names from M1, M2 and M3 columns into one column
	mat_impact_df['one_mat_name'] = mat_impact_df['M3'].copy() #[hard coding for column name]
	for col_name in ['M2', 'M1']:
		mat_impact_df['one_mat_name'].fillna(mat_impact_df[col_name], inplace=True)
	# combine the mat name with 'location', 'unit'  --> to match the format in the incoming  mat_name_match_dict
	mat_impact_df['combined_act'] = mat_impact_df.apply(lambda row: ' | '.join([str(row['one_mat_name']),str(row['location']),str(row['unit'])]), axis=1) 


	""" *** extract the material impacts of interest from the entire mat_impact_df *** """
	matched_mat_from_impact_dict = {}
	# match activities in the bmat_name_match_dict with the activities in the mat_impact_df
	#print(f"size of mat_name_match_dict is: {len(mat_name_match_dict)} \n") # this show how many BoM materials are included (virgin +  recycled counterpart)
	for matched_name in set(mat_name_match_dict.values()):
		try:
			impact_only_df = mat_impact_df.loc[mat_impact_df['combined_act']==str(matched_name)] # will return an empty dataframe if the condition is not met
		except KeyError as e:
			# log the warning
			logger.info(" =================================================================== ")
			logger.info(f"[Error] {e} -> == {matched_name} == does not have a match in the impact factors dataset, skip to next BoM material \n")
			logger.info(" =================================================================== ")
			continue # skip rest of this iteration

		# catch all exceptions and skip rest of this iteration
		if len(impact_only_df) == 0: 
			# log the warning
			logger.info(" =================================================================== ")
			logger.info(f"[WARNING!!!] == {matched_name} == is not in M1, M2 or M3 column, something's wrong...")
			#[Caution] if there is no M1, M2 or M3 name for that material, it will NOT be included in the matched_mat_from_impact_dict below
			logger.info(f"[CAUTION] == {matched_name} == is NOT included in matched_mat_from_impact_dict")
			logger.info(" =================================================================== ")
			continue

		# extract impact results: 
		impact_only_df = impact_only_df.iloc[:,6:-2] # [hard coding warning] last two columns are 'one_mat_name' and 'combined_act'
		matched_mat_from_impact_dict[matched_name] = impact_only_df.values

	""" *** quantify the total impacts *** """
	# convert matched impact results dict to dataframe
	matched_impact_df = pd.DataFrame.from_dict([matched_mat_from_impact_dict])
	# get # of impact categories investigated
	n_lcia = len(matched_impact_df.iloc[0,0][0])

	# if there is 'nan' for one or more of impact values in the matched df, drop those materials
	matched_impact_df.dropna(axis=1, inplace=True)

	# if there is 'nan' for one or more of the bom mat categories in archetype, replace it with zero
	slice_bom_quantity_df = slice_bom_quantity_df.astype("float") # for unknown reason, some of the numeric values in the archetype_BoM_df are 'object' dtype, so need to force the type to 'float'
	slice_bom_quantity_df = slice_bom_quantity_df.fillna(0)

	# multiply the dataframe of BoM with matched impact dataframe (units should be all matched at this point)
	tmp_df_list = []
	for bom_name, act_name in mat_name_match_dict.items():
		try:
			tmp_results = slice_bom_quantity_df[bom_name].apply(lambda x: np.multiply(x,matched_impact_df[act_name]))
			tmp_results = pd.DataFrame(tmp_results).set_axis([bom_name],axis=1)
		except KeyError as e: # the act_name of mat_name_match_dict still contains 'no matching is found' (which does not exist in matched_impact_df)
			tmp_results = slice_bom_quantity_df[bom_name].apply(lambda x: [np.repeat('no value due no to matching',n_lcia)])
		tmp_df_list.append(tmp_results)
	total_impacts_by_mat = pd.concat(tmp_df_list,axis=1)

	return total_impacts_by_mat, impact_category_idx_list


## define a function to save results to a local folder
def store_results(save_name: str, storage_path: str, results_to_store_list: list, impact_category_idx_tuple_list: list, recipe: pd.DataFrame) -> None:
	"""
	This functions stores the modeling results into a .xlsx file to a local folder

	Input params:
	- save_name: name of the excel file to save, str
	- storage_path: path to the destination folder for storage, str
	- results_to_store_list: a list contains the dataframes of total impact (can be more than one) from each material (col) of interest for archetypes (row) for each recipe, list
	- impact_category_idx_tuple_list: a list of (idx, impact_category_name) of the corrseponding impact category results in the df, list
	- recipe: contains the recipe information of a building archetype, pd.DataFrame 
	"""

	# combine storage path with file name
	storage_path = os.path.sep.join([storage_path,'.'.join([save_name,'xlsx'])])

	# create the ExcelWriter obj
	writer = pd.ExcelWriter(storage_path, engine='xlsxwriter')

	# loop over the list of dfs
	for idx, df in enumerate(results_to_store_list):
		# make a copy
		restore_df = df.copy()
		df = df.copy()
		# identify the impact category of interest
		for impact_category_idx_tuple in impact_category_idx_tuple_list:
			for mat_header in df.columns:
				df[mat_header] = df[mat_header].apply(lambda x: x[0][impact_category_idx_tuple[0]]) # identify the corresponding impact category result using the idx from impact_category_idx_tuple
				# save to sheet of a given results df with a specific impact category
				# add a new column of impact category name
			df['impact assessment method'] = impact_category_idx_tuple[1]
			# concate the recipe df and clean up
			df = pd.concat([df,recipe], axis=1)
			for col in recipe.columns:
				df[col] = df[col].fillna(df[col].loc[0]) #'0' is an index label in this case
			df = df.drop(0) # 0-index where NaN for all mat columns but values for recipe attributes (created after concate)
			df.to_excel(writer, sheet_name=f'sheet_{idx}_{impact_category_idx_tuple[0]}')
			# restore df to its original content
			df = restore_df.copy()

	# save to all sheets to the excel file
	# format: # of sheets = # of df x # of impact categories
	writer.save()


## define a function to merge result sheets
def merge_sheets(storage_path: str, path_to_sheets: str, one_sheet_name: str, impact_category_dict: dict, **kwargs) -> None:
	"""
	This function loads the .xlsx sheets of interest, merge and reformat them into a new .xlsx sheet. Current format for the resulting .xlsx sheets:
	- named by 'multiple mapping retained_x', where 'x' refers to the rank (0 being the hightest score for matching between BoM name and impact factor name)
	- each sheet contain # of tabs: each tab contains the results for a specific impact category
	- within each tab: rows -> all building archetypes from a given database, cols -> BoM materials + attributes of recipe template (e.g., Occupation, building_type)

	Input params:
	- storage_path: path to the destination folder for storage, str
	- path_to_sheets: path to the folder from which sheets to be loaded, str
	- one_sheet_name: name of a the resulting single sheet, str
	- impact_category_dict: {idx:impact_category_name}, dict
	- kwargs
	  * file_name_retained: a string to retained the files of interest, str

	"""

	# initialize variables
	loaded_sheets_dict = defaultdict(list) # {lcia_method: [df1,df2,...]}

	# create the ExcelWriter obj
	writer = pd.ExcelWriter(os.path.sep.join([storage_path, '.'.join([one_sheet_name,'xlsx'])]),engine='xlsxwriter')

	# [thanks ChatGPT!]
	# Get a list of all the excel files in the folder 
	# excel_files = [f.name if f.is_file() else f for f in os.scandir(path_to_sheets)]
	excel_files = [f for f in os.scandir(path_to_sheets) if os.path.splitext(f)[1] == '.xlsx'] # this line does work for files in google drive, as the extentions are not included in file name


	# retain only the files of interest
	if 'file_name_retained' in kwargs:
		excel_files = [os.path.splitext(os.path.basename(f))[0] for f in excel_files if all(kw in os.path.splitext(os.path.basename(f))[0] for kw in kwargs['file_name_retained'])]

	# Loop through each excel file and read all the sheets in it
	for f_name in excel_files:
		# Use a context manager to open the excel file
		try: 
			with pd.ExcelFile(os.path.sep.join([path_to_sheets, ".".join([f_name,'xlsx'])]),engine='openpyxl') as xls: # ".".join([f_name,'xlsx'])
				# Loop through each sheet (containing results for individual impact categories) in the file
				for sheet_name in xls.sheet_names:
					# Load the data from the sheet into a DataFrame: row: building ID, col headers: BoM material names, impact category name, attributes of recipe template
					loaded_sheets_dict[impact_category_dict[int(sheet_name[-1])]].append(pd.read_excel(xls, sheet_name=sheet_name)) # last element of sheet name is the idx that correspond to the impact assessment method in the dict
																																	# e.g., 'sheet_0_1' -> last element is '1' which correspond to {1: ('ReCiPe Midpoint (H) V1.13 no LT', 'water depletion', 'WDP')}
		except Exception as e:
			# log the error msg
			logger.info(" =================================================================== ")
			logger.info(f"[ERROR] The following error occurred: '{e}', '{f_name}' is skipped!!")
			logger.info(" =================================================================== ")
			# [caution] for unknown reason, even if the exception is raised, the df is still added to the loaded_sheets_dict -> causing weird formatting & repeated values in 1st sheet of the merged file

	# concat the dfs into one 
	for lcia_name, df_list in loaded_sheets_dict.items():
		# clean up the lcia_name: ('ReCiPe Midpoint (H) V1.13 no LT', 'climate change', 'GWP100')
		lcia_name = lcia_name[1:-1] # remove the "()"
		lcia_name = [part_name.replace("'","") for part_name in lcia_name.split(',')] # remove qutation marks
		lcia_name = [part_name.replace(".","_") for part_name in lcia_name] # replace . with _
		lcia_name = "_".join([lcia_name[0][:5],lcia_name[0][-5:],lcia_name[-1]])

		one_df = pd.concat(df_list)
		one_df.to_excel(writer,sheet_name=lcia_name)
  	
	# save as new xlsx file
	writer.save()


## define a function to process results for generating plots
def process_for_plot(raw_df: pd.DataFrame, mat_cols: list, scenario_info: dict) -> pd.DataFrame:
	"""
	this function prepares the data for plots

	Input params:
	- raw_df: lcia sheet of interest, DataFrame
	- mat_cols: column names that correspond to material names, list
	- scenario_info: {'scenario description': condition}, dict

	Returns:
	- df_to_plot: processed data ready for plot, DataFrame
	"""
	# initiate the output
	df_to_plot = raw_df.copy()

	# change "Unnamed: 0" to "building_ID"
	df_to_plot.rename(columns={"Unnamed: 0": "building_ID"},inplace=True)
	# calculate total impacts
	df_to_plot['TOTAL impacts'] = df_to_plot[mat_cols].apply(lambda x: pd.to_numeric(x,errors='coerce')).sum(skipna=True, axis=1)
	# add scenario info: e.g., 'RC' (i.e., change recycled content): 0.2
	df_to_plot[list(scenario_info.keys())[0]] = list(scenario_info.values())[0]

	return df_to_plot

## define a wrapper
def impact_calc_wrapper_manual(recipe: pd.DataFrame, bom_df_processed: pd.DataFrame, mat_impact_df: pd.DataFrame, mat_name_match_dict: dict,
                                **kwargs) -> (pd.DataFrame,list,list):
	"""
	This function performs the LCA calculation for a single recipe (WB-BoM, can be multiple buildings that all meet the recipe criteria):
	- generate archetype of interest
	- apply strategies (optional)
	- calculate the LCIA results
	Input params
	- recipe: contains the recipe information of a building archetype, pd.DataFrame
	- bom_df_processed: a dataframe contains the archetype classification and corresponding BoM (preprocessed), pd.DataFrame
		*currently support:
		- Heeren & Fishman 2018
	- mat_impact_df: a dataframe contains material names (by different specification lvl) and corresponding impact factors (e.g., kg CO2-eq/unit), pd.DataFrame
	- mat_name_match_dict: a dict contains the {bom_mat_name: name_in_mat_impact_sheet | location | unit}, dict
	- kwargs
		* strategy_dict: a dict contains {'strategy_name': list of dict_strategy_implementation}
		* unit_convert_dict: contains the converstion factors for units of bom (e.g., from kg to m3 for concrete), {bom_name: unit_convert_factor}, dict
			- these factors should be used as mutipliers
  	Returns:
    - total_impacts_by_mat: a dataframe contains the total impact from each material (col) of interest for archetypes (row)
    - archetype_labels_list: a list of archetype labels that are later used to label the results in visualization (from each recipe), list
      	* used to be a nested list (when multiple recipes are processed in this function) -> now it's the same as archetype_labels output from archetype_gen() func
    - impact_category_idx_list: a list of (idx, impact_category_name) of the corrseponding impact category results in the df to plot
  """

	# generate the archetype of interest
	archetype_BoM_df,archetype_idx,archetype_labels_list = archetype_gen(bom_df_processed,recipe)

	# check if need to apply strategies
	if 'strategy_dict' in kwargs:
		# apply strategies
		archetype_BoM_df = apply_strategies(archetype_BoM_df,kwargs['strategy_dict'])

	# perform unit conversion, if needed
	if 'unit_convert_dict' in kwargs:
		# perform the calculation and append to the output
		total_impacts_by_mat,impact_category_idx_list = calc_impact_multi(mat_name_match_dict,mat_impact_df,archetype_BoM_df, unit_convert_dict=kwargs['unit_convert_dict'])
	else:
		# perform the calculation and append to the output
		total_impacts_by_mat,impact_category_idx_list = calc_impact_multi(mat_name_match_dict,mat_impact_df,archetype_BoM_df)
  
	return total_impacts_by_mat, archetype_labels_list, impact_category_idx_list

