# Multivariate Pattern Analysis Code

### Introduction
Project investigating whether perceptions of facial attractiveness can be predicted from patterns of activity in the Orbito Frontal Cortex and the Nucleus Accumbens. 

The experimental feature of average attractiveness was obtained from ratings of facial stimuli from a large independent pool of subjects (~1000) and an individual rating from a sample of subjects participating in a face perception task. The ratings of the first category are hereafter referred to as ‘average attractiveness’, and those from the second category as ‘subjective attractiveness’ (participant specific).

### Pattern estimation and preprocessing
The data were analyzed using multivariate pattern analysis. More specifically, we ran a regression analysis to predict average and subjective attractiveness using ridge regression, which computes the linear least squares. The data were already preprocessed using the fMRIprep preprocessing pipeline (Esteban et al., 2019). The pattern estimation had already been done with the ‘Least-squares all’ (LSA) technique, implemented by NiBetaSeries (Brett, et al., 2018). To load in these patterns, we wrote a function looking for the file of each run. In the same function, all runs among the two sessions were concatenated per subject. 
> Example_R01=get_R('sub-01')

> print(Example_R01.shape)

The following ROIs were defined using the Harvard-Oxford Atlas (Kennedy, & Haselgrove, 2011): the Frontal Orbital Cortex, the left nucleus accumbens, and the right nucleus accumbens. Each ROI was resampled to the neural data of each subject to match in resolution. Below an example of how this would be done for the Frontal Orbital Cortex using the above extracted R of subject 1.
> Example_FOC01=get_R_roi('Frontal Orbital Cortex',Example_R01)

> print(Example_FOC01.shape)

We additionally applied a pattern drift filter function. Pattern drift is the autocorrelation between sequential trials, which, if not corrected, can cause additional noise in the data. We removed the low-frequency pattern drift from each voxel using a polynomial basis set. The function for this was written specifically for the whole dataset. The function we use to apply this filter runs through each run for each ROI.

### Machine Learning Pipeline
A linear regression analysis was run, predicting both average and subjective attractiveness separately from each ROI. The pipeline included a `StandardScaler` from the sklearn python package (Pedregosa et al., 2011), which standardizes the features with mean centering and scales to unit variance, while a `Ridge` regression was used as the loss function. We adopted a `LeaveOneGroupOut` cross-validation strategy: where folds were created based on different runs. Through each iteration, the model is fit on all trials but those of one run, with this run then being used as the test set. Using this strategy, we computed the (average) R squared for each ROI for subjective and average attractiveness.
> linear_regr

### Model performance and statistical evaluation
Now we have all our variables in order to run a linear regression analysis (LRA). LRA was run for each ROI twice, once predicting the average attractiveness ratings and once predicting the subjective attractiveness rating. R squared was calculated to evaluate the model performance. In our regression, R squared shows the **porportion** of variance in ratings explained by the pattern activity.

We chose the R squared as a measure of performance because it computes the variance accounted for and has a mean chance level of zero. The computed R squared is therefore better inpertretable than the mean squared error.

Pearson correlations were calculated between the subjective and average ratings for each subject. 

In order to investigate whether the proportion of explained variance differed significantly within ROIs for different attractive ratings, we performed a paired sample t-test between the subjective and average R2 scores.

For the ROIs that showed a significant difference between explained variance, Cohen’s d was calculated to assess the effect size. Cohen's d was calculated by dividing the mean difference by the pooled standard deviation.

_This code was co-created with Nada Amekran and Ava Ma De Sousa Pernes_

### References
- Brett, M., Hanke, M., Markiewicz, C., Côté, M. A., McCarthy, P., Ghosh, S., & Wassermann, D. (2018). nipy/nibabel: 2.3. 0. June. https://doi.org/10.5281/zenodo.1287921
- Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A., ... Oya, H. (2019). fMRIPrep: A robust preprocessing pipeline for functional MRI. Nature methods, 16(1), 111-116.
- Kennedy, D., & Haselgrove, C. (2011). Harvard-oxford atlas
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
