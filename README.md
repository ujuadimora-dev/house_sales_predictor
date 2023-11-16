
# Housing Pricing Issues

**Data Analysis and Predictive Modelling Study**

**Developed by: Adimora Uju Louisa**

![Moch_site](media/moch_site.png)

**Live Site:** [Live webpage](https://housing-sales-price-predictor-0ea2e265945b.herokuapp.com/)

**Link to Repository:** [Repository](https://github.com/ujuadimora-dev/house_sales_predictor)

## Table of Content

- [Housing Pricing Issues](#housing-pricing-issues)
  - [Table of Content](#table-of-content)
  - [Introduction](#introduction)
  - [CRISP-DM Workflow](#crisp-dm-workflow)
  - [Business Requirements](#business-requirements)
  - [Dataset Content](#dataset-content)
  - [Hypothesis, proposed validation and actual validation](#hypothesis-proposed-validation-and-actual-validation)
  - [Mapping the business requirements to the Data Visualisations and ML tasks](#mapping-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
  - [ML Business Case](#ml-business-case)
    - [Predict Sale Price](#predict-sale-price)
  - [Dashboard Design](#dashboard-design)
    - [Page 1: Project Summary](#page-1-project-summary)
    - [Page 2: Sale Price  Analysis](#page-2-sale-price--analysis)
    - [Page 3: Sale Price Prediction](#page-3-sale-price-prediction)
    - [Page 4: Project Hypothesis and Validation](#page-4-project-hypothesis-and-validation)
    - [Page 5: Machine Learning Model](#page-5-machine-learning-model)
  - [Unfixed Bugs](#unfixed-bugs)
  - [PEP8 Compliance Testing](#pep8-compliance-testing)
  - [Deployment](#deployment)
    - [Heroku](#heroku)
  - [Technologies](#technologies)
    - [Development and Deployment](#development-and-deployment)
    - [Main Data Analysis and Machine Learning](#main-data-analysis-and-machine-learning)
  - [Credits](#credits)
    - [Sources of code](#sources-of-code)
    - [Media](#media)
  - [Acknowledgements](#acknowledgements)

## Introduction


The  House Price Sale Predictor site is  Maching Learning model  and Data Analysis that  applied to a real estate managining of Houses (Ames, Iowa) and it is used to predict the sale value of a property based on certain features of the home. It also allows the user to see how certain features of a home correlate with the sale price of the home. The site is made of five(5) main sections, summary section, Project hypothesis, sales price predictor, ML and valuation and sales price analysis


## CRISP-DM Workflow

The project was developed using the Cross Industry Standard Process for Data Mining. This follows several iterations over well defined steps:

1. **Epic 1: Business Understanding** - This incorporates an extensive discussion with the client and their expectations as well as the development of acceptance criteria. These are layed out in the [Business Requirements](#business-requirements) below.
2. **Epic 2: Data Understanding** - The data needed to achieve the business requirements must be identified and understood. Are the data available to answer the business requirements? An initial statistical analysis helps to determine whether the data available are adequate to answer the business requirements. This task is carried out in the Data Cleaning Notebook.
3. **Epic 3: Data Preparation** - Clean and impute data, carry out feature engineering, such as transformations or scaling, reformat data if needed. This step is very important to ensure the most effective and accurate modelling outcome. This is carried out in the Data Cleaning and Feature Engineering Notebooks.
4. **Epic 4: Modelling** - Determine the model algorithms to be used. Split the data into train and test sets. Use train sets to validate various algorithms and tune them using a hyperparamter search. This is carried out in the Model_Evalution_Regression_1 Notebook.
5. **Epic 5: Evaluation** - Use the test set to evaluate the model performance. Match these with the business acceptance criteria.  This is carried out in the Model_Evalution_Regression_1 and Model_Evalution_lRegression_2 Notebooks.
6. **Epic 6: Deployment** - Develop the streamlit app that will satisfy the business requirements determined in collaboration with the client and deploy the app online. The app is deployed in Heroku and the process is described in the [Deployment](#deployment) section below.

These steps can be matched up nicely to 6 Epics in the Agile development process. As we move along the pipeline of the development process we may flow back and forth between stages/epics as we learn new insight and have to revisit previous step in order to refine the development. While ultimately moving towards the final delivery of a product that satisfies the users/clients requirements.

## Business Requirements

A client who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, has requested help in maximising the sales price for the inherited properties.

The client has an excellent understanding of property prices in her own state and residential area, but she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and has provided it for this project.

The client is interested in the following outcomes:

1. Discovering how the house attributes correlate with the sale price and which attributes most affect the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price for illustration.
  
2. Predicting the house sale price from her four inherited houses and any other house in Ames, Iowa, based on the most important features of the homes. The predictive model should aim to achive an R2 value of 0.8 or higher.

3. Delivery of the final product in the form of a deployed app that is easily accessible online and userfriendly.  

These requirements can also be viewed as the user stories of the client/end user.

1. **User Story 1**: As a client, I want to be able to discover how features of a home correlate with the sale price, so that I can gain insight into the importance of a homes features in determining the sale price.
2. **User Story 2**: As a client, I want to be able to determine the likely sale price of a home based on certain features, so that I can gain insight into the likely values of a given home in the area.
3. **User Story 3**: As a client, I want to be able to access the required information  online, so that, i will enable me to access it anythime and anywhere.

User Stories - Data Practitioner:
From the project requirements, we can create a list of user stories for either a data practioner or standard non-technical user.

4. **User Story 4**:As a data practitioner, I want to import a public dataset into the system so that I can build a model to predict the sales price of the inherited houses located in Ames, Iowa.

5. **User Story 5**:As a data practitioner, I want to clean and process the dataset so that I can build an accurate model for predicting house prices.

6. **User Story 6**:As a data practitioner, I want to explore the dataset to understand the features and their relationships with the sale price so that I can create valuable visualizations.

7. **User Story 7**:As a data practitioner, I want to build a predictive model that accurately predicts the sale price of the inherited properties as well as any other house in Ames, Iowa.

8. **User Story 8**:As a data practitioner, I want to optimize the model's hyperparameters to ensure that it is as accurate as possible to the clients expected requirements.

9. **User Story 9**:As a data practitioner, I want to test the model's efficiency and accuracy and ensure that it is reliable for achieving our needs.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). The business requirements are based on a fictitious, but realistic, user story described above. Predictive analytics can be applied here in a real world scenario.

- The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built, etc.) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Hypothesis, proposed validation and actual validation

In order to fullfill the business requirements and in discussion with the client, the following hypotheses have been developed:

1. We hypothesize that a property's sale price correlates strongly with a subset of the extensive features in the dataset. We aim to validate this using a correlation study of the dataset.
   - The extensive correlation study we carried out and displayed on the app, confirms this hypothesis.
2. We hypothesize that the correlation is strongest with common features of a home, such as total square footage, overall condition and overall quality. We aim to validate this using a correlation study.
   - The extensive correlation study confirms that the five features with the strongest correlation to Sale Price are: 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', and '1stFlrSF'. These are all features common to the majority of the homes.
3. We hypothesize that we are able to predict a sale price with an R2 value of at least 0.8. We propose to validate this by developing a predictive model, optimizing it using data modelling tools and evaluating it based on the required criteria.
   - The model evaluation has validated this hypothesis and in fact we were able to achieve R2 values of 0.84 and more for both test and train sets.

## Mapping the business requirements to the Data Visualisations and ML tasks

- Business Requirement 1: Data Visualization and Correlation Study.
  - We will inspect the data contained in the data set as it relates to the property sale prices in Ames, Iowa.
  - We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to the sale price.
  - We will plot the most important and relevant data against the sale price to visualize the insights.

- Business Requirement 2: Regression and Data Analysis
  - We want to predict the sale price of homes in Ames, Iowa. For this purpose we will build a regression model with the sale price as the target value.
  - We will carry out optimization and evaluation steps in order to acchieve an R2 value of 0.8 or higher.

- Business Requirement 3: Online App and Deployement
  - We will build an app using streamlit that displays all the desired data analysis and visualization as well as a feature that will allow the client to predict the sale prices for her and any other property in Ames, Iowa.
  - We will deploy the app using Heroku.

## ML Business Case

### Predict Sale Price

- We want an ML model to predict sale price, in dollars, for a home in Ames, Iowa. The target variable is a continuous number. We firstly consider a regression model, which is supervised and uni-dimensional.
- Our ideal outcome is to provide a client with the ability to reliably predict the sale price of any home in Ames, Iowa, and more specifically the inherited properties the client is particularly concerned with.
- The model success metrics are:
  - At least 0.8 for R2 score, on train and test set.
  - The model is considered a failure if: after 12 months of usage, the model predictions are 50% off more than 30% of the time, and/or the R2 score is less than 0.8.
- The output is defined as a continuous value of sale price in dollars. Private parties/home owners/clients can access the app online and input data for their homes. The app can also be useful for real estate agents who want to give a quick estimate of saleprice to a prospective client, they can input the data on the fly while in live communication with a prospective client.
- The training data come from a public data set, which contains approx. 1500 property sales records. It contains one target features: sale price, and all other variables (23 of them) are considered features.

## Dashboard Design

The project will be built using a Streamlit dashboard. The completed dashboard will satisfy the third Business Requrirement. It will contain the following pages:

### Page 1: Project Summary

This page will incude:

- Statement of the project purpose.
- Brief description of the data set.
- Statement of business requirements.
- Links to further information.

<details>
<summary>Project Summary Page Screenshots</summary>
<img src="media/summary_page.png" width="60%">
<img src="media/summary_page2.png" width="60%">
</details>

### Page 2: Sale Price  Analysis

This page will fullfill the first business requirement. It includes checkboxes so the client has the ability to display the following visual guides to the data features:

- A sample of data from the data set.
- Pearson and spearman correlation plots between the features and the sale price.
- Histogram and scatterplots of the most important predictive features.
- Predictive Power Score analysis.

<details>
<summary>Sales Analysis Screenshots</summary>
<img src="media/sales_price_analysis1.png" width="60%">
<img src="media/sales_price_analysis2.png" width="60%">
<img src="media/sales_price_analysis3.png" width="60%">
<img src="media/sales_price_analysis4.png" width="60%">
<img src="media/sales_price_analysis5.png" width="60%">
<img src="media/sales_price_analysis6.png" width="60%">
<img src="media/sales_price_analysis7.png" width="60%">
<img src="media/sales_price_analysis8.png" width="60%">
<img src="media/sales_price_analysis9.png" width="60%">
<img src="media/sales_price_analysis10.png" width="60%">
<img src="media/sales_price_analysis11.png" width="60%">
</details>

### Page 3: Sale Price Prediction

This page will satisfy the second Business Requirement. It will include:

- Input feature of property attributes to produce a prediction on the sale price.
- Display of the predicted sale price.
- Feature to predict the sale prices of the clients specific data in relation to her inherited properties.

<details>
<summary>Sale Price Predictor Screenshots</summary>
<img src="media/price_prediction_1.png" width="60%">
<img src="media/price_prediction_2.png" width="60%">

</details>

### Page 4: Project Hypothesis and Validation

This page will include:

- A list of the project's hypothesis and how they were validated.

<details>
<summary>Project Hypothesis and Validation Screenshots</summary>
<img src="media/project_hypo.png" width="60%">
</details>

### Page 5: Machine Learning Model

This page will include

- Information on the ML pipeline used to train the model.
- Demonstration of feature importance.
- Review of the pipeline performance.

<details>
<summary>ML Screenshots</summary>
<img src="media/ml_1.png" width="60%">
<img src="media/ml_2.png" width="60%">
<img src="media/ml_3.png" width="60%">
</details>

## Unfixed Bugs

The app does not currently contain any unfixed bugs. 

## PEP8 Compliance Testing

All python files where passed through the [CI Python Linter](https://pep8ci.herokuapp.com/). Those files incuded the `app_pages` files and the files in the `src` folder. A few small errors were fixed, such as long lines or trailing white spaces. Finally, no errors were detected.

## Deployment

### Heroku

- The App live link is: <housing-sales-price-predictor-0ea2e265945b.herokuapp.com>
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Technologies

This section contains information on resources and technologies used to complete this project.

### Development and Deployment

- [GitHub](https://github.com/) was used to create the project repository, story project files and record commits.
- [Code Anywhere](https://codeanywhere.com/) was used as the development environment.
- [Jupyter Notebooks](https://jupyter.org/) were used to analyse and engineer the data, and develop and evaluate the model pipeline.
  - In the terminal type `jupyter notebook --NotebookApp.token='' --NotebookApp.password=''` to start the jupyter server.
- [Heroku](https://www.heroku.com/) was used to deploy the project.
- [Kaggle](https://www.kaggle.com/) was used to access the dataset
- [Streamlit](https://streamlit.io/) was used to develop Dachbord

### Main Data Analysis and Machine Learning

- [NumPy](https://numpy.org/) was used for mathematical operations for examples determining means, modes, and standard deviations.
- [Pandas](https://pandas.pydata.org/) was used for reading and writing data files, inspecting, creating and manipulating series and dataframes.
- [ydata_profiling](https://ydata-profiling.ydata.ai/docs/master/index.html) was used to create an extensive Profile Report of the dataset.
- [PPScore](https://pypi.org/project/ppscore/) was used to determine the predictive power score of the data features.
- [MatPlotLib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) were used for constructing plots to visualize the data analysis, specifically the heatmaps, correlation plots and historgram of feature importance.
- [Feature Engine](https://feature-engine.trainindata.com/en/latest/index.html) was used for various data cleaning and preparation tasks:
  - Dropping Features, and Imputation of missing variables.
  - Ordinal Encoding, Numerical Transformations, Assessment of outliers, and Smart Correlation Assessment of variables.
- [SciKit Learn](https://scikit-learn.org/stable/) was used for many machine learning tasks:
  - splitting train and test sets.
  - feature processing and selection.
  - gridsearch to determine the optimal regression model.
  - gridsearch to determine the optimal hyperparameters.
  - evaluation  of the model using r2_score.
  - Principal Component Analysis and evaluation.
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) for the XGBoostRegressor algorithm.

## Credits

### Sources of code

- The CI Churnometer Walkthrough Project  was used to source various functions and classes in the development process, such as: HyperparameterOptimizationSearch, Feature Importance analysis, evaluation of train and test sets, PPS and Correlation Analysis and plots, Missing Data Evaluation, Data Cleaning Effect, etc. These are all used in the Jupyter Notebooks during the development process of the project.
- The CI Churnometer Walkthrough Project was also the source of the Steamlit pages which were then modified and adapted to the app deployed in this project.
- More generally, The walkthrough project provided a guide for the general layout and flow of the project.

### Media

- The image of house Ames is from Microsoft Power Point images


## Acknowledgements

Many thanks and appreciation go to the following sources;

- Several past projects provided valuable additional information on how to complete a successful project:
  - Heritage Housing Issues project by T. Hullis [Link](https://github.com/t-hullis/milestone-project-heritage-housing-issues)
  - Heritage Housing Issues project by Ulrike Riemenschneider [Link](https://github.com/URiem/heritage-housing-PP5)
- The Slack community has, as always, been invaluable in answering questions.  resolve several technical issues.
- StackOverflow helped resolve several issues through out the project.
