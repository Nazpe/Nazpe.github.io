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

<alt="image" src="https://github.com/user-attachments/assets/4e9057a1-e620-478e-a7a6-a9b0ebd0f06d" />{: .mx-auto.d-block :}




My name is Inigo Montoya. I have the following qualities:

- I rock a great mustache
- I'm extremely loyal to my family

What else do you need?

### My story

To be honest, I'm having some trouble remembering right now, so why don't you just watch [my movie](https://en.wikipedia.org/wiki/The_Princess_Bride_%28film%29) and it will answer **all** your questions.
