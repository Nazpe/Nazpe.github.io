---
layout: post
title: Mortality Prediction in ICU Patients with Pneumonia
subtitle: A small overview of what I discovered
gh-repo: nazpe/Thesis
gh-badge: [star, follow]
tags: [LR]
comments: true
mathjax: true
author: Nuno Pedrosa
---

{: .box-note}
This section is a simple summary of my academic work in Mortality prediction via logistic regression in ICU patients with pneumonia.

{: .box-note}
I wrote and presented an article based on this work to the CIARP 2023 conference (26th Iberoamerican Congress on Pattern Recognition). [The paper can be found in the proceedings of the conference, here.](https://link.springer.com/chapter/10.1007/978-3-031-49249-5_3)

## Introduction

Pneumonia is a very relevant problem with repercussions for society. In 2019, it resulted in 2.5 million deaths worldwide as well as causing a significant loss of quality of life and financial costs. Therefore, the prediction of pneumonia-related mortality is critical, and this work explores machine learning to address it. Literature has already made based on e.g. Random Forests and XGBoost. 

Although the use of complex models can result in good predictions, the use of simpler and interpretable models, such as logistic regression, enables to better understand the rules and correlations of variables in a model. Another important aspect that is often overlooked is the significant variation that exists within the populations from which the data originates. Different subsets of the population possess unique characteristics that can have varying impacts on mortality predictions. To address this, one approach is to partition the data into more homogeneous populations and develop separate models for each population group.

Accordingly, this work focuses on the development of interpretable models, that enable drawing meaningful conclusions from the data and gaining (clinical) insights into the underlying process. This work further explores localized prediction.

## Materials

The data under analysis contains information on 15355 admissions in the ICU diagnosed with pneumonia, in Portugal, from February 02, 2009 to August 18, 2020. This analysis extracted information in the 24-48h time window after admission of the patients into ICU. After pre-processing, the sample consisted of a set of 64 features from 2729 admissions.

![Extraction Pipeline](https://github.com/user-attachments/assets/7c03023f-aca5-458f-8324-3acbea5b2372){: .mx-auto.d-block :}

**Fig. 1.** Microsoft SQL Server Database Pipeline, with SQL Based procedures, Data Extraction and Data Preparation 


## Methods

#### Data Pre-Processing

![Data Pre-Processing](https://github.com/user-attachments/assets/4e9057a1-e620-478e-a7a6-a9b0ebd0f06d){: .mx-auto.d-block :}

**Fig. 2.** Data Pre-Processing Pipeline, in transformation, data was transformed with the Yeo-Johnson family of transformations, in selection, a correlation dendrogram analysis was performed based on dissimilarity to eliminate features.

#### Predictive Models  

Twelve different models were created, 2 global models, denoted by M24-48PS and OSM24-48PS and 10 localized models, the M24-48PSC family and the OSM24-48PSC family. In the names of the models, M24-48PS, stands for mortality given the 24-48h patient status, the OS, in the beginning, for oversampled data and the C, in the end, for clustered.

![Predictive Models](https://github.com/user-attachments/assets/76b898f7-da9c-489a-bebb-0d9d910a7355){: .mx-auto.d-block :}

**Fig. 3.** Pipeline to obtain the models. Blue squares are in common for all models, green squares are just for the OS model and orange squares are just for C models.

Important aspects of the pipeline include: 
* Use of **Logistic Regression** and Recursive Feature Elimination (RFE) to choose the most important features for the final models.
* Oversampling via **ADASYN** to deal with unbalanced data.
* **Hierarchical clustering** dendrogram analysis was used to separate the data into populations and optimal separation indicated 5 clusters.

## Results and Discussion

![Predictive Models](https://github.com/user-attachments/assets/001a7ede-3cbe-4419-b3fa-e606a7c4d4f2){: .mx-auto.d-block :}

**Fig. 4.** Models Summary, with the features used in each model with their coefficient value, number of train observations used and the mortality ratios.

Some relevant aspects that can be highlighted from Fig. 4. are the great variety of different mortality ratios in the different clusters, evidencing that different populations in the data have different probabilities of mortality. The great variation in the number of observations for each population. And the fact that each cluster had an optimal pull of different features shows that important factors to predict mortality vary from population to population.

To compare global and localized results, two systems were created: 
* **Membership Separation (M)**: The observations of the test data are assigned to the cluster with a smaller distance to the cluster centroid.
* **Via Weights (W)**: All test data is predicted using all models, and the final probability for a given observation is the weighted average over all model predictions depending on cluster centroids distances.

![Predictive Models](https://github.com/user-attachments/assets/f37f2668-0b4c-4e00-8ef4-b21240d5e832){: .mx-auto.d-block :}

**Fig. 5.** Balanced accuracy of the models. Other metrics were also calculated (Accuracy, Precision, Recall, F1 Score, AUROC, and the optimal separation threshold), but they were not analyzed here for the sake of simplicity in this overview.

The localized and global approach's reveal approximately the same performance, probably because there isn’t a big enough heterogeneity in our data to lead to better predictions in the localized models. Comparing OS models to the not OS models, the OS models perform better in cross validation but worse in test.

#### Localized Models Proof of Concept

 To show that in a dataset with greater separability, localized models can give
 better performances, the centroids of the previously obtained clusters were taken
 and the 75 closest observations for train and the 50 closest test observations were
 extracted. Three new models were created, NEW 3, NEW 4 and NEW 5. Cluster
 1 and 2 models, because of the already low amount of data present, were used
 as before.

 For comparison reasons, a global model with new data with greater
 separability was created (NEW GLOBAL). A new silhouette score between all clusters of 0.2, evidencing a larger separability than the one obtained before (0.11).

The **new global model and the weighted average of the new localized models** got **0.60 ± 0.13, 0.67 ± 0.20** in cross validation and **0.76, 0.77** in test, respectively, in balanced accuracy. The results may indicate that localized models would be useful in more separable data, but aren’t enough for a definitive answer.

## Conclusions

It was possible to obtain mortality predictions of patients with pneumonia in the 4 different approaches, standing out M24-48PS and M24-48PSC W. There was no evident advantage of the localized models, due to the low separability of the data, however, the proof of concept showed that it as potential in more separable data.

Future work will explore different approaches aiming to increase the performance of the predictive models, including optimizing data pre-processing parameters, as well as clustering analysis, and considering other clinical variables (or other codifications of the existing ones). It will also experiment with more complex models that permit better performances. Lastly, it would be important to continue this work together with professionals in the field of health, who can provide insights on how to tune the models according to their needs. Overall, it is very important that this research for optimizing the prediction of mortality in pneumonia is continued, in an effort to improve health care.
