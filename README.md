# Statistical Methods for Data Science — Group E Project

## Overview

This project analyzes an insurance dataset to predict whether existing health-insurance clients are interested in purchasing vehicle insurance. We compare GLM (logistic regression), GAM, and Random Forest models, focusing on interpretability, parsimony (Occam’s razor), and out-of-sample performance. 

## Authors: A. Gottardi, A. Minutolo, E. Corrolezzis, L.F. Palacios Flores. 


## Problem Statement

An insurance company wants to identify policyholders who may also be interested in vehicle insurance to optimize outreach and revenue. The prediction target is Response (1 = interested, 0 = not interested). 

## Data

We consider the provided train.csv (with target), while test.csv lacks the Response variable and sample.csv is not useful for modeling; therefore, only train.csv is analyzed. Key variables include demographics (Gender, Age, Region_Code), policy details (Annual_Premium, Policy_Sales_Channel, Vintage), and vehicle info (Vehicle_Age, Vehicle_Damage). 

## During EDA:

- The id field is removed (uninformative). 

- We visualize categorical variables via bar plots and numerical variables via densities (in the Rmd). 

- Feature Engineering (high level)

- Reduced factors: original high-cardinality fields are consolidated as Channels_Reduced and Region_Reduced for modeling. 

- Vintage is later dropped based on AIC/ANOVA results (low utility). 

## Modeling Approach

We use a static 70/30 train–test split and fit/compare three model families. 

1) GLM (Logistic Regression)

- Stepwise AIC (MASS::stepAIC) starting from a full model (direction = "both") to study variable utility. Findings: Vintage is dispensable; core signals include Previously_Insured, Vehicle_Damage, Age, outreach Channels_Reduced, Vehicle_Age, and Region_Reduced. 

- Nested models: custom ranking removing one variable at a time (AIC), then building up parsimonious models guided by Occam’s razor; ANOVA (Chisq) confirms the variable order and diminishing returns beyond Region_Reduced.

- Interpretation highlights (GLM):

- Prior vehicle insurance drastically decreases the odds of interest (Previously_Insured = Yes).

- Past Vehicle_Damage strongly increases interest.

- Age shows a mild negative effect; Vehicle_Age (older cars) increases interest; Channels_Reduced and Region_Reduced matter; Annual_Premium has negligible effect.
Detailed coefficient summaries are in the Rmd/HTML.

2) GAM (mgcv)

- We replicate the nested-model ranking with splines for Age (and optionally Annual_Premium), improving AIC vs. linear age.

- The optimal GAM (balance of AIC and parsimony) uses Previously_Insured, Vehicle_Damage, s(Age), Channels_Reduced, Region_Reduced.

- Plots show categorical effects and the smooth for Age. 

3) Random Forest

- Due to compute limits, we sample 60% of the reduced training set and fit 500 trees, using caret for partitioning and pROC for ROC/AUC & threshold selection (Youden’s J). Class imbalance yields high error on the minority class without threshold tuning. 

- Variable importance (Mean Decrease Gini) ranks Vintage high for node purity despite its weak GLM/GAM role—a common tree-based behavior. Vehicle_Damage is less critical here than in GLM/GAM.

- Performance (held-out test), comparing best models per family:

GLM: best Accuracy, FPR, TNR, Precision.

GAM: best AUC, TPR, FNR (and competitive Precision).

Random Forest: slightly behind but comparable overall.

Full metric table is in the pdf report. 

## Conclusions

The GAM provides the best overall explanation–performance trade-off with Previously_Insured, Vehicle_Damage, s(Age), Channels_Reduced, Region_Reduced; adding more variables complicates the model without meaningful performance gains.
