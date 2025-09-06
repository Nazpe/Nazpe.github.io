---
layout: post
title: Mortality Prediction in ICU Patients with Pneumonia
subtitle: A small overview of what I discovered
gh-repo: nazpe/Thesis
gh-badge: [star, follow]
tags: [test]
comments: true
mathjax: true
author: Nuno Pedrosa
---

{: .box-success}
I wrote and presented an article based on this work to the CIARP 2023 conference (26th Iberoamerican Congress on Pattern Recognition). [The paper can be found in the proceedings of the conference, here.](https://link.springer.com/chapter/10.1007/978-3-031-49249-5_3)

## Introduction

Pneumonia is a very relevant problem with repercussions for society. In 2019, it resulted in 2.5 million deaths worldwide as well as causing a significant loss of quality of life and financial costs. Therefore, the prediction of pneumonia-related mortality is critical, and this work explores machine learning to address it. Literature has already made based on e.g. Random Forests and XGBoost. This work focuses on the development of interpretable models, that enable drawing meaningful conclusions from the data and gaining (clinical) insights into the underlying process. This work further explores localized prediction.

## Materials

The data under analysis contains information on 15355 admissions in the ICU diagnosed with pneumonia, in Portugal, from February 02, 2009 to August 18, 2020. This analysis extracted information in the 24-48h time window after admission of the patients into ICU. After pre-processing, the sample consisted of a set of 64 features from 2729 admissions.

## Methods

#### Data Pre-Processing

<img width="627" height="426" alt="image" src="https://github.com/user-attachments/assets/4e9057a1-e620-478e-a7a6-a9b0ebd0f06d" />{: .mx-auto.d-block :}

![Benjamin Bannekat](https://github.com/user-attachments/assets/4e9057a1-e620-478e-a7a6-a9b0ebd0f06d){: .mx-auto.d-block :}

**Fig. 1.** Data Pre-Processing Pipeline, in transformation, data was transformed with the Yeo-Johnson family of transformations.

#### Predictive Models  

Twelve different models were created, 2 global models, denoted by M24-48PS and OSM24-48PS and 10 localized models, the M24-48PSC family and the OSM24-48PSC family. In the names of the models, M24-48PS, stands for mortality given the 24-48h patient status, the OS, in the beginning, for oversampled data and the C, in the end, for clustered.

<img width="654.5" height="294.5" alt="image" src="https://github.com/user-attachments/assets/b15383f8-a1e9-4859-992d-199482087928" />{: .mx-auto.d-block :}

**Fig. 2.** Pipeline to obtain the models. Blue squares are in common for all models, green squares are just for the OS model and orange squares are just for C models.

Important aspects of the pipeline include:
Use of Logistic Regression and Recursive Feature Elimination (RFE) to choose the most important features for the final models.
Oversampling via ADASYN to deal with unbalanced data.
Hierarchical clustering dendrogram analysis was used to separate the data into populations and optimal separation indicated 5 clusters.
<img width="3949" height="380" alt="image" src="https://github.com/user-attachments/assets/4a1a1d55-56bc-4a38-92ce-a03d4cab8ecf" />



My name is Inigo Montoya. I have the following qualities:

- I rock a great mustache
- I'm extremely loyal to my family

What else do you need?

### My story

To be honest, I'm having some trouble remembering right now, so why don't you just watch [my movie](https://en.wikipedia.org/wiki/The_Princess_Bride_%28film%29) and it will answer **all** your questions.
