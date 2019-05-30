# How_to_win_data-science_competition-Final_project


**Competition** : Coursera Final Project- Future Sales prediction challenge

**Name** : Sumit Bhattacharya

 -------------------------------------------------------------------------------------------------------------------------------------
 
## 1. Summary
I had to face difficulty in the beginning understanding the data. I tried to extract text
features from the datasets by
a) Converting the Text to English using Translate package in Python. But that did not
give me any proper information regarding the dataset. The only information I coul
gather using this was that item categories id which were nearby in the item
category datasets tend to belong to similar categories.
b) Using Tfidf TextVectorizer. It improved the score a bit, but was not satisfactory.
So, I did not use Text Features.
Then , EDA helped me to conclude that per month sales were decreasing as average price
was increasing. So using these as prima facie, I did some task as described below in other
sections.

## 2. Feature Selection / Extraction
Important features were related to the number of sales in recent months.
Mean Encodings have also been used for categorical features. Lag Features were
created. Due to insufficient hardware resources, lags for only 3 months are created.
Kaggle Kernels informed me that maximum lag was for 12 months. There were many
outliers in the training dataset which have been taken care of.

## 3. Training Methods
I trained 3 models.

### a) Catboost :- 
My first aim was to make use of CatBoost Library as it was taught as an
optional lecture in the Coursera course. CatBoost is a very powerful and fast
Boosting Library for Boosted trees. It is much faster than XGBoost and LightGBM.
And using the GPU of Colab , it expedited a lot. CatBoost works best with categorical
features.
But contrary to my expectations, this did not provide sufficiently good result. Even
after optimizing the model with “hyperopt” package the minimum rmse on test data
was 1.002. As CatBoost works best with categorical features, and in my data
wrangling I got rid of all the textual data and the so-called categorical IDs of items
and shops are numeric in nature with a huge range. So maybe I could not leverage
the ultimate advantage of CatBoost. For using CatBoost, my data should have been
prepared in another way taking into account categorical features.

### b) LightGBM:- 
Using LightGBM is better than XGBOOST due to speed and memory
overhead issues. The only problem using LightGBM is that Colab has to be
configured properly to set up GPU for LightGBM. On the other hand, XGBOOST can
directly comply with GPU settings once runtime is set to GPU type. Gradient
Boosting using LightGBM can be efficiently reliable.


### c) RandomForest:-
RandomForest is one of the best algorithms available. It is a
highly customizable model that can run easily on a GPU or a CPU(using parallel
construction of tress).


## 4. Model Execution Time
RandomForest took about 25 min to train and 1 min to predict. LightGBM took about 40
min to train and 1 min to predict. I did all my work at Google Colab. The GPU Runtime
provides TESLA K80 GPU with compute capacity of 3.7, GPU RAM of 12.72 GB, and GPU
Memory 358.27 GB.

### Dependencies

**numpy** 1.16.3

**pandas** 0.24.2

**sklearn** 0.21.2

**lightgbm** 2.2.3

**matplotlib** 3.0.3

**catboost** 0.15
