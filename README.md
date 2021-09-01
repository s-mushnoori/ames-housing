# ames-housing
Price prediction on the Ames housing [dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

---
**Note:** This readme will guide you through the steps I took to complete this project. To view the final refactored script, please take a look at `ames-housing.py`

---

# Table of Contents
1. `00_DataPrep.ipynb`
    1. Preprocessor class

2. `01_EDA.ipynb` 

3. `02_FeatureEngineering.ipynb`
    1. Feature generation
    2. Feature transformation
    3. Feature encoding
    4. Feature selection
    5. EngineerFeatures class

4. 

---
## 1. `00_DataPrep.ipynb`

First we explore the data and come up with a strategy to deal with missing values. 

We note the following details about the features:
1. **We have 'regular' categorical features where a null value represents a missing value**
    * These will be imputed with the _mode_ of the feature

2. **We have 'special' categorical features where a null value carries a meaning as described in the `data_description.txt` file in the dataset**
    * These will be imputed with the string `'None'` which will represent the meaning carried by the null value

3. **We have 'regular' numeric features where a null value represents a missing value**
    * These will be imputed with the _median_ of the feature

4. **We have 'special' numeric features which are tied to other features. For example, if a house does not have a basement, the basement surface area will be 0**
    *   The null values for these features will be imputed with `0` 

5. **Finally, we note that `'MSSubClass'` is actually a categorical feature. This will be converted to a `str` type and treated as a categorical variable during feature engineering**

With this strategy in mind, we create a `Preprocessor` class where a `.fit()` method calculates the modes and medians based on the training set and a `.transform()` method imputes the null values with the appropriate values. 

---
## 2. `01_EDA.ipynb`

* **Categorical features such as the neighborhood affecting sale price:**
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/Neighborhood.PNG">

* **Numeric features such as basement surface area, and living room surface area correlate strongly with the sale price:**
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/BasementSF.PNG">
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/LivSF.PNG">

* **More houses are sold in the summer, peaking in June. However, the sale price is not strongly correlated to the month of sale:**
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/MoSold.PNG">

* **A particularly important observation is that several features are skewed. This will be taken into account during feature engineering:**
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/LotAreaSkew.PNG" width=440>
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/BasementUnfSkew.PNG" width=440>
<img  src="https://github.com/s-mushnoori/ames-housing/blob/main/Images/1stFlrSkew.PNG" width=440>

---
## 3. `02_FeatureEngineering.ipynb`

### 3.1 Feature generation
asdasda alksjdla

---


