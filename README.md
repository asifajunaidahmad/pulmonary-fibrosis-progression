# pulmonary-fibrosis-progression
Forecasting the decline in lung function (FVC, measured by spirometry) of patients with pulmonary fibrosis from sparse, irregular follow-up visits, and quantifying the uncertainty of each forecast.
# Problem

For each patient, given a baseline CT scan, clinical metadata (age, sex, smoking status) and a baseline FVC, predict FVC at future weeks together with a confidence value (sigma). Submissions are scored with a modified Laplace log-likelihood (higher is better; sigma is clipped at 70 and errors at 1000):

score = -sqrt(2) * delta / sigma_clipped - ln(sqrt(2) * sigma_clipped)
# Data

Downloaded from Kaggle

# Item                          Value

Training set	                  1,549 FVC measurements from 176 patients
Measurements per patient	      mostly 9 (132 patients); range 6 to 10
Test set	                      5 patients (baseline visit only)
FVC (training)	                mean 2,690, std 833, range 827 to 6,399
Features	                      Weeks (relative to baseline CT), FVC, Percent, Age, Sex, SmokingStatus
Images	                        DICOM chest CT per patient

# What is in the notebook

Code has three parts:\n

1. Exploratory analysis: FVC distribution, per-patient trajectories, feature interactions by sex and smoking status, correlations. The only strong correlation is between FVC and Percent (Percent is derived from FVC).\
2.Two-stage ElasticNet: pairs every visit with every other visit of the same patient (baseline visit + weeks passed), predicts FVC with an ElasticNet (alpha=0.3, l1_ratio=0.8), then predicts the per-row confidence with a second ElasticNet. Cross-validation is a 7-fold GroupKFold by patient.\
2.Quantile neural network: a small dense network (two hidden layers of 100 units) on tabular features that outputs the 0.2, 0.5 and 0.8 quantiles, trained with a blend of pinball loss and the competition metric, 7-fold CV. The code also defines a DICOM loading function, but it is not called, so this model uses tabular features only despite the "CNN" label in the notebook.

# Results

# Model	                                        Metric	                                       Value
ElasticNet, fixed sigma = 100 	        Laplace log-likelihood (CV)	                          -7.135
ElasticNet, predicted sigma	            Laplace log-likelihood (CV)	                          -6.731
ElasticNet, per-row optimal sigma	      Laplace log-likelihood	                              -6.337   
Quantile NN	                            MAE of median FVC	                                    139.9 mL
Quantile NN	                            Mean predicted interval width (0.2 to 0.8)            239.1 mL

The headline number is the ElasticNet with predicted sigma (-6.731). The quantile NN figures are reported for completeness and should be read with the caveats below.

# Limitations 
Quantile NN validation is likely optimistic. It uses the Percent column of the same visit as the FVC target. Percent is computed from FVC, so this appears to leak the target during training and validation, while at test time only the baseline Percent is available. It also uses plain KFold, which can place visits of the same patient in both training and validation folds. The ElasticNet pipeline drops Percent and uses GroupKFold by patient.\n
Very small cohort: 176 training patients and 5 test patients, so cross-validation estimates have high variance.\n
No imaging yet: none of the models uses the CT scans, so this is a tabular baseline for a multimodal problem.\n
No subgroup analysis yet: performance by sex, age and smoking status has not been evaluated.\n

# Roadmap
Fix the NN evaluation: remove same-visit Percent, switch to GroupKFold by patient, report the Laplace metric for the NN.\
Extract CT features (pretrained 2D/3D CNN or lung-segmentation features) and fuse them with the tabular data.\
Missing-modality ablation: compare tabular-only, image-only and fused models, and evaluate when the CT is unavailable at test time.\
Subgroup evaluation of error and calibration by sex, age band and smoking status.\
Per-patient progression models (e.g. hierarchical Bayesian slope models) as a stronger baseline for sparse longitudinal data.\

# Acknowledgements and sources
Competition and data: OSIC and Kaggle.

