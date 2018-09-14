
import numpy as np
import os,glob
import random

#### function to create a subsample from main dataset #######
def dat_subsample(csvpath,sample_size=2000):
	### return list of images that were selected, list of images not selected 
	### the dataframe of image selected and the dataframe of images not selected
	pneumonia = 8964
	notpeunomia = 11500+8525
	total_im = pneumonia+notpeunomia
	#putting a seed number so that everyone has the same list of random numbers 
	random.seed(1023)
	info_file = pd.read_csv(csvpath+'stage_1_detailed_class_info.csv')
	#selecting the 3 classes and saving them in different dataframes
	pneu_df = info_file[(info_file['class']=='Lung Opacity')]
	notnorm_df = info_file[(info_file['class']=='No Lung Opacity / Not Normal')]
	norm_df = info_file[(info_file['class']=='Normal')]
	# creating 3 random number list
	#pneu_ranlst = nprand.randint(0,pneu_df.shape[0],sample_size//2)
	#notnorm_ranlst = nprand.randint(0,notnorm_df.shape[0],sample_size//4)
	#norm_ranlst = nprand.randint(0,norm_df.shape[0],sample_size//4)
	pneu_ranlst = random.sample(range(pneu_df.shape[0]),sample_size//2)
	notnorm_ranlst = random.sample(range(notnorm_df.shape[0]),sample_size//4)
	norm_ranlst = random.sample(range(norm_df.shape[0]),sample_size//4)
	#randomly selected image dataframe 
	sel_pneu_lst = pneu_df.iloc[pneu_ranlst].index.values.tolist()
	sel_notnorm_lst = notnorm_df.iloc[notnorm_ranlst].index.values.tolist()
	sel_norm_lst = norm_df.iloc[norm_ranlst].index.values.tolist()
	#joining all the list to get a complete list of images
	main_sel_lst = []
	main_sel_lst.extend(sel_pneu_lst)
	main_sel_lst.extend(sel_notnorm_lst)
	main_sel_lst.extend(sel_norm_lst)

	#getting the list of selected images and getting the corresponding dataframe
	selected_image_lst = info_file.iloc[main_sel_lst]['patientId'].values.tolist()
	selected_image_df = info_file.iloc[main_sel_lst]

	#getting the not selected list and dataframe - could be use for cross validation
	notselected_image_df = info_file.loc[info_file.index.difference(selected_image_df.index), ]
	notselected_image_lst = notselected_image_df['patientId'].values.tolist()
	return selected_image_lst,notselected_image_lst,selected_image_df,notselected_image_df
