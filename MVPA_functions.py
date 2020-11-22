#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Function to get R per subject

def get_R(sub): #input as 'sub-17'
    data_dir = get_data_dir() #define directory 
    #preallocate
    R_all=[]
    files=[] 
    for ii in range(2): #loop over sessions
        session = 'ses-'+str(ii+1) #session as 'ses-1' or 'ses-2' for filenames        
        pattern_dir = os.path.join(data_dir, 'derivatives', 'nibetaseries_lsa_unn', sub, session, 'func')
        search_str = os.path.join(pattern_dir, '*MNI152*STIM*betaseries.nii.gz')            
        files_allruns = sorted(glob(search_str))       
        files.append(files_allruns) #append files 
    ## Combine .nii files of both session per subject
    R_all=image.concat_imgs(files)
    return R_all # returns all images concatenated for all sessions & runs; for 1 subject


# In[3]:


# Function to extract  either the subject or average actractiveness ratings per subject

def get_S(sub,rating): #sub input as 'sub-01', rating input as 'average' OR 'subject'
    ### ---- ###         
    return S_all #returns all ratings per participant; len(S_all) should be 480


# In[4]:


# Function to get R for the specific region of interest (either Frontal orbital cortex, or right/left 
# nucleus accumbens)
def get_R_roi(ROI, R): 
    from nilearn.datasets import fetch_atlas_harvard_oxford
    if ROI == 'Frontal Orbital Cortex': #Cortical atlas for Frontorbital cortex
        ho_atlas=fetch_atlas_harvard_oxford('cort-maxprob-thr50-2mm')
    else: #Subortical atlas for left and right accumbens
        ho_atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')      
    
    #Define ROI, incl resampling 
    ho_map = nib.load(ho_atlas['maps'])
    f_ROI_indx = ho_atlas['labels'].index(ROI)
    ho_map_resamp = image.resample_to_img(ho_map, R, interpolation='nearest')  
    f_ROI_bool = ho_map_resamp.get_fdata() == f_ROI_indx
    f_ROI_roi = nib.Nifti1Image(f_ROI_bool.astype(int), affine=ho_map_resamp.affine)
    R_roi = masking.apply_mask(R, f_ROI_roi)    
    return R_roi


# In[5]:


# Function for plotting ROI of subject 1 in the introduction
# only used for illustratory reasons
def showlocation(ROI): 
    from nilearn.datasets import fetch_atlas_harvard_oxford
    if ROI == 'Frontal Orbital Cortex': #Cortical atlas for Frontorbital cortex
        ho_atlas=fetch_atlas_harvard_oxford('cort-maxprob-thr50-2mm')
    else: #Subortical atlas for left and right accumbens
        ho_atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')    
      
    R = get_R('sub-01')
    
    #Define ROI, incl resampling 
    ho_map = nib.load(ho_atlas['maps'])
    f_ROI_indx = ho_atlas['labels'].index(ROI)
    ho_map_resamp = image.resample_to_img(ho_map, R, interpolation='nearest')
    
    f_ROI_bool = ho_map_resamp.get_fdata() == f_ROI_indx
    f_ROI_roi = nib.Nifti1Image(f_ROI_bool.astype(int), affine=ho_map_resamp.affine)
    plotroi = plotting.plot_roi(f_ROI_roi);

    return plotroi


# In[6]:


# Function to run linear regression and calculate Rsquared per participant per rating per ROI
def linear_regr(R_roi, S_all):
    # Number of stimuli per run
    N_per_run=40
    # Number of runs per session
    n_run=6
    # Number of sessions per participant
    n_ses=2
    
    #Create pipeline with leave one group out cross validation
    scaler = StandardScaler()
    ridge = Ridge()
    pipe = make_pipeline(scaler, ridge)    
    logo = LeaveOneGroupOut()
   
    groups =  np.concatenate([[i] * N_per_run for i in range(n_run*n_ses)])

    folds = logo.split(R_roi, S_all, groups)
    
    #preallocate Rsquared

    r_squares = []

    # Run pipeline over folds
    for i, fold in enumerate(folds):
        train_idx, test_idx = fold
        
        #Fit the model on train set
        pipe.fit(R_roi[train_idx], S_all[train_idx])
        
        #Predict test set
        preds = pipe.predict(R_roi[test_idx])

        # compute cross-validated R squared 
        Rsquared=r2_score(S_all[test_idx],preds)
        
        #store R squared
        r_squares.append(Rsquared)
        
    #Compute mean R2 per participant per region
    avg_R2= np.mean(r_squares)
    
    return avg_R2


# In[7]:


# Function that applies pattern drift function and filters images per run
def get_R_filt(R_rois):

    R_all_filt = []
    for pattern_subj in R_rois:
        #initialize R_filt_subj as an np array of shape (0, 650) or shape (0, 26)
        R_filt_subj = np.zeros((0, pattern_subj.shape[1]))
        for i in range(0, pattern_subj.shape[0]-1, 40):

            #extract individual runs
            R_run = pattern_subj[i: i+40, :]

            #apply filter to runs
            R_run_filt = filter_pattern_drift(R_run, deg=8)

            #add filtered run to the subject's array
            R_filt_subj = np.concatenate((R_filt_subj, R_run_filt), axis=0)

        #append np subject's array to the R_all_filt list, which includes all subjects
        R_all_filt.append(R_filt_subj)
        
    return(R_all_filt)


# In[8]:


def example_R_filt(R_rois):
    R_all_filt=[]
    R_filt_subj=np.zeros((0, R_rois.shape[1]))
    for i in range(0, R_rois.shape[0]-1, 40):

            #extract individual runs
            R_run = R_rois[i: i+40, :]

            #apply filter to runs
            R_run_filt = filter_pattern_drift(R_run, deg=8)

            #add filtered run to the subject's array
            R_filt_subj = np.concatenate((R_filt_subj, R_run_filt), axis=0)

    #append np subject's array to the R_all_filt list, which includes all subjects
    R_all_filt.append(R_filt_subj)
        
    return(R_all_filt)

