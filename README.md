# House Price Predictor
The **House Price Predictor** is an interactive dashboard for analyzing housing data from Ames, Iowa, and predicting house sale prices. This project combines data analysis, machine learning, and interactive visualizations to provide actionable insights for homeowners, real estate agents, and property investors.

By following a structured methodology ([CRISP-DM](#crisp-dm-and-machine-learning-business-case)) and using advanced machine learning techniques, this project delivers accurate predictions and valuable insights tailored to the needs of its users.

[**Try the Dashboard Here**](https://pp5-sales-price-predictor.onrender.com)

![Am I responsive screenshot](docs/readme-imgs/house-price-predictor.png)

## Project Summary
This project analyzes housing data from Ames, Iowa, to predict house sale prices using machine learning. The key findings include:
- **Strongest Predictors**: Overall quality (`OverallQual`) and living area (`GrLivArea`) are the most influential factors.
- **Model Performance**: The regression model achieved an R² score of 0.793 on the test set.
- **Interactive Dashboard**: Users can explore data, validate hypotheses, and predict house prices in real-time.

## Table of Contents
1. [**House Price Predictor**](#house-price-predictor)
	- [Project Summary](#project-summary)
2. [**Terminology**](#terminology)
3. [**CRISP-DM and Machine Learning Business Case**](#crisp-dm-and-machine-learning-business-case)
   - [Summary of CRISP-DM Phases and Their Purpose](#summary-of-crisp-dm-phases-and-their-purpose)
   - [Detailed Explanation of Each Phase](#detailed-explanation-of-each-phase)
4. [**Dataset Content**](#dataset-content)
5. [**Business and Dashboard Requirements**](#business-and-dashboard-requirements)
6. [**Data Exploration and Cleaning**](#data-exploration-and-cleaning)
7. [**Hypotheses and Validation Process**](#hypotheses-and-validation-process)
	- [Hypotheses and Results](#hypotheses-and-results)
	- [Key Insights from Correlations Analysis](#key-insights-from-correlation-analysis)
8. [**Agile Methodology: EPICS and User Stories**](#epics-and-user-stories)
9. [**Technical Implementation of Business Requirements**](#technical-implementation-of-business-requirements)
10. [**Dashboard Design**](#dashboard-design)
	- [Why We Chose the New Streamlit Approach](#why-we-chose-the-new-streamlit-approach)
	- [Sidebar and Page Descriptions](#sidebar-and-page-descriptions)
11. [**Plots**](#plots)
12. [**Testing**](#testing)
    - [Testing Overview](#testing-overview)
    - [Code Quality Testing with PEP8](#code-quality-testing-with-pep8)
    - [Functional Testing](#functional-testing)
    - [Responsive Testing](#responsive-testing)
    - [EPICS and User Stories Testing](#epics-and-user-stories-testing)
    - [Model Unit Testing](#model-unit-testing)
    - [Jupyter Notebook Testing](#jupyter-notebook-testing)
13. [**Bugs**](#bugs)
    - [Bugs Identified During Development](#bugs-identified-during-development)
	- [Bugs Identified After Deployment](#bugs-identified-after-deployment)
14. [**Future Improvements**](#future-improvements)
15. [**Deployment**](#deployment)
16. [**Technologies and Python Packages**](#technologies-and-python-packages)
17. [**Credits**](#credits)
	- [Content](#content)
	- [Media](#media)
18. [**Acknowledgements**](#acknowledgements)

## Terminology
This section provides definitions for key terms and concepts used throughout the project to ensure clarity and accessibility for all readers.

- **CRISP-DM**: A structured process model for data mining projects, consisting of six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- **R² (R-squared)**: A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Higher values indicate better model performance.
- **MAE (Mean Absolute Error)**: A metric used to measure the average magnitude of errors in a set of predictions, without considering their direction.
- **MSE (Mean Squared Error)**: A metric that measures the average squared difference between predicted and actual values. It penalizes larger errors more than smaller ones.
- **RMSE (Root Mean Squared Error)**: The square root of the MSE, providing an interpretable measure of error in the same units as the target variable.
- **Feature Engineering**: The process of transforming raw data into features that better represent the underlying problem to the predictive models, improving their performance.
- **Hyperparameter Tuning**: The process of optimizing the parameters of a machine learning model to improve its performance.
- **Streamlit**: An open-source Python library used to create interactive web applications for data science and machine learning projects.
- **Correlation**: A statistical measure that describes the strength and direction of a relationship between two variables.
- **Heatmap**: A graphical representation of data where individual values are represented by varying colors, often used to visualize correlations.
- **Regression Model**: A type of predictive modeling technique that estimates the relationship between a dependent variable (target) and one or more independent variables (features).
- **Pearson Correlation**: A measure of the linear relationship between two variables, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- **Spearman Correlation**: A non-parametric measure of rank correlation, assessing how well the relationship between two variables can be described using a monotonic function.
- **Predictive Power Score (PPS)**: A metric that quantifies the predictive strength of one variable for another, regardless of their linear or non-linear relationship.
- **Winsorization**: A statistical technique used to limit extreme values in the data to reduce the effect of outliers.
- **Dashboard**: An interactive interface that provides visualizations, insights, and tools for users to explore data and generate predictions.

## CRISP-DM and Machine Learning Business Case
This project follows the **CRISP-DM methodology** (CRoss Industry Standard Process for Data Mining) to ensure a structured and systematic approach to solving the client's business problems. The methodology was applied as follows:

### Summary of CRISP-DM Phases and Their Purpose
The table below provides a high-level overview of the six CRISP-DM phases and their purpose in this project:

| **Phase**              | **Explanation**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| **Business Understanding** | Ensures the project is meets the client's goals and provides a clear roadmap for the analysis. |
| **Data Understanding**  | Focuses on exploring and understanding the dataset to identify key variables and patterns. |
| **Data Preparation**    | Involves cleaning, transforming, and preparing the data for analysis and modeling. |
| **Modeling**            | Builds and optimizes machine learning models to meet the business requirements. |
| **Evaluation**          | Assesses the model’s performance to ensure it meets the client's expectations. |
| **Deployment**          | Delivers the final solution to the client, including a user-friendly dashboard. |


### Detailed Explanation of Each Phase
#### 1. Business Understanding
- **Objective**: Address the client's needs:
  1. Identify which house attributes correlate most strongly with sale prices.
  2. Predict the sale prices of four inherited houses and other properties in Ames, Iowa.
- **Outcome**: Two main tasks were defined:
  - Perform data analysis and visualization to identify correlations.
  - Build a machine learning model to predict house prices.
#### 2. Data Understanding
- **Dataset**: Sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data), containing 79 variables describing house attributes and sale prices.
- **Key Variables**: Features like `GrLivArea`, `OverallQual`, and `GarageArea` were identified as potentially significant predictors.
#### 3. Data Preparation
- **Steps**:
  - Handled missing values (e.g., imputation for `LotFrontage`).
  - Created new features like `TotalSF` (Total Square Footage) and `HouseAge`.
  - Addressed outliers and optimized data types.
- **Outcome**: A clean dataset ready for analysis and modeling.
#### 4. Modeling
- **Model**: An **ExtraTreesRegressor** was selected for its ability to handle both linear and non-linear relationships.
- **Performance**:
  - **Train Set R²**: 0.809
  - **Test Set R²**: 0.793
- **Key Features**: `OverallQual`, `GrLivArea`, `GarageArea`, and `KitchenQual`.
#### 5. Evaluation
- The model exceeded the client's expectations, achieving an R² score above the required 0.75 on the test set.
- Visualizations like residual plots and feature importance charts were used to validate the model's performance.
#### 6. Deployment
- A user-friendly dashboard was developed using **Streamlit** to present insights and allow real-time predictions.
- The app was deployed via [Render](https://render.com) for easy access.

By combining data analysis, machine learning, and an interactive dashboard, this project delivers actionable insights and accurate predictions tailored to the client's needs.

## Dataset Content
- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). A fictitious user story was created to demonstrate how predictive analytics could be applied in a real workplace project.
- The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating the house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

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

## Business and Dashboard Requirements
This project's success depends on meeting the client's business and dashboard requirements. These requirements ensure that the solution is both actionable and user-friendly, addressing the client's need to maximize the value of their inherited properties in Ames, Iowa.

By addressing these requirements, the project delivers a comprehensive solution that combines data analysis, machine learning, and an interactive dashboard to meet the client's goals.

- **Business Requirement 1**: The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualizations of the correlated variables against the sale price to show that.
- **Business Requirement 2**: The client is interested in predicting the house sale price from her four inherited houses and any other houses in Ames, Iowa.
- **Dashboard Requirements**: The client requires a dashboard that provides an overview of the project and dataset, highlights key correlations between house attributes and sale prices, and allows users to input house data for real-time price predictions. It should also display the predicted sale price for the 4 inherited houses, their total value, and include a technical page summarizing the model's performance and pipeline.


## Data Exploration and Cleaning
Before conducting the analysis and building the predictive model, the dataset was thoroughly explored and cleaned to ensure data quality and reliability. This step was crucial for addressing missing values, optimizing data types, and preparing the dataset for hypothesis validation and model training.

#### Key Steps in Data Cleaning:
1. **Handling Missing Values**:  
   - Missing values were analyzed and handled based on their impact on the dataset.  
   - Features with a high percentage of missing values (e.g., `EnclosedPorch`, `WoodDeckSF`) were dropped due to their limited contribution to the analysis.  
   - For other features, missing values were imputed using appropriate strategies, such as filling with the median (`LotFrontage`) or mode (`GarageType`).
2. **Optimizing Data Types**:  
   - Data types were optimized to reduce memory usage and improve processing speed.  
   - For example, categorical variables like `MSZoning` and `Neighborhood` were converted to the `category` data type.
3. **Feature Engineering**:  
   - New features were created to enhance the predictive power of the model.  
   - For instance, `TotalSF` (Total Square Footage) was introduced by combining `TotalBsmtSF`, `1stFlrSF`, and `2ndFlrSF`.  
   - Temporal features like `HouseAge` and `RemodelAge` were derived from `YearBuilt` and `YearRemodAdd`.
4. **Outlier Detection and Removal**:  
   - Outliers in key numerical features (e.g., `GrLivArea`, `SalePrice`) were identified using visualizations like scatter plots and box plots.  
   - Extreme outliers were removed to improve model performance and reduce skewness.
5. **Encoding Categorical Variables**:  
   - Categorical variables were encoded to make them suitable for machine learning models.  
   - Techniques like one-hot encoding and label encoding were applied to variables such as `Neighborhood` and `GarageType`.
6. **Saving the Cleaned Dataset**:  
   - The cleaned dataset was saved as a CSV file for further analysis and modeling.

#### Conclusion:
The data cleaning process ensured that the dataset was free of inconsistencies, optimized for analysis, and ready for hypothesis validation and model training. This step laid the foundation for accurate and reliable predictions.

## Hypotheses and Validation Process
To address **Business Requirement 1**, we formulated and tested four hypotheses to explore how house attributes affect sale prices. These hypotheses were validated using correlation analysis and visualizations, which provided insights into the most significant predictors of house prices.

### Hypotheses and Results
#### Hypothesis 1: Larger houses have a higher sale price.
- **Rationale**: It is expected that houses with larger living area (`GrLivArea`, `GarageArea`, `TotalBsmtSF`) have higher prices due to their size and usability.
- **Validation**: 
	- A correlation analysis showed that `GrLivArea` has a strong positive correlation with sales price *(Pearson correlation: 0.71)*.
	- A scatterplot showed a clear trend where larger living area resulted in higher prices.
- **Results**: The hypothesis was validated. `GrLivArea` is one of the most significant variables in predicting sales price.
- **Visualization**:

![GrLivArea vs SalePrice](docs/plots/lm_plot_price_by_GrLivArea.png)
	
#### Hypothesis 2: Houses with higher overall quality have a higher sale price.
- **Rationale**: Houses with better construction quality and finish (`OverallQual`) are expected to have higher prices due to their durability, aesthetics and buyer appeal.
- **Validation**: 
	- A correlation analysis showed that `OverallQual` has a very strong positive correlation with sales price *(Pearson correlation: 0.79)*.
	- A boxplot showed that houses with higher construction quality consistently had higher prices.
- **Results**: The hypothesis was validated. `OverallQual` is one of the most decisive factors for the sales price.
- **Visualization**:

![OverallQual vs SalePrice](docs/plots/box_plot_price_by_OverallQual.png)

#### Hypothesis 3: Newer houses have a higher sale price.
- **Rationale**: Newer houses (`YearBuilt`) are expected to have higher prices due to modern design, better materials and lower maintenance costs.
- **Validation**: 
	- A correlation analysis showed a positive correlation between `YearBuilt` and sales price *(Pearson correlation: 0.52)*.
	- A line plot showed that newer houses generally have higher prices.
- **Results**: The hypothesis was partially validated. Although newer houses have higher prices, the correlation is not as strong as for other variables.
- **Visualization**:

![YearBuilt vs SalePrice](docs/plots/line_plot_price_by_YearBuilt.png)

#### Hypothesis 4: Houses with garages have a higher sale price.
- **Rationale**: Houses with garages (`GarageArea`) are more attractive because they offer extra storage and parking space, which increases value.
- **Validation**: 
	- A scatterplot showed a positive trend between `GarageArea` and sales price *(Pearson correlation: 0.62)*.
	- Houses with larger garage areas generally had higher prices.
- **Results**: The hypothesis was validated. `GarageArea` affects sales price, but not as strongly as `GrLivArea` or `OverallQual`.
- **Visualization**:

![GarageArea vs SalePrice](docs/plots/lm_plot_price_by_GarageArea.png)

### Key Insights from Correlation Analysis
- **Strongest Predictors**: `OverallQual` and `GrLivArea` are the most influential variables, with strong positive correlations to `SalePrice`.
- **Moderate Predictors**: `GarageArea` and `YearBuilt` also impact sale prices but to a lesser extent.


## EPICS and User Stories
This project was developed by following a structured approach based on Epics and User Stories. Each Epic represents a major phase of the project, while the User Stories break down the tasks into actionable steps. All Epics and User Stories have been addressed throughout the project to ensure that the client's requirements are met.

You can find the detailed tracking and progress of these Epics and User Stories on GitHub: [Project Board on GitHub](https://github.com/users/NatashaRy/projects/4)

### **EPIC 1 - Information gathering and data collection**
- User Story (E1US01) - **As a data analyst**, I want to download the dataset from Kaggle so that I can start analyzing it.
- User Story (E1US02) - **As a data analyst**, I want to explore the structure of the dataset so that I can identify key variables. 
- User Story (E1US03) - **As a data analyst**, I want to document the business requirements so that I can ensure the project meets the client's need. 
 
### **EPIC 2 - Data visualization, cleaning and preparation**
- User story (E2US04) - **As a data analyst**, I want to clean the dataset by handling missing values and outliers so that it is ready for analysis.
- User story (E2US05) - **As a data analyst**, I want to perform a correlation analysis so that I can identify which variables have the most impact on the sale price. *(Business Requirement 1, Dashboard Requirements)*
- User story (E2US06) - **As a data scientist**, I want to create visualization of the correlations so that I can present insights to the user. *(Business Requirement 1, Dashboard Requirement)*

### **EPIC 3 - Model training, optimization and validation**
- User story (E3US07) - **As a data analyst**, I want to split the dataset into training and testing data so that I can evaluate the model's performance. 
- User story (E3US08) - **As a data analyst**, I want to build a regression model so that I can predict house prices. *(Business Requirement 2, Dashboard Requirement)*
- User story (E3US09) - **As a data analyst**, I want to optimize the model using hyperparameter tuning so that I can improve its performance. *(Business Requirement 2, Dashboard Requirement)*
- User story (E3US10) - **As a data analyst**, I want to evaluate the model using R² so that I can ensure it meets the performance requirements. *(Business Requirement 2, Dashboard Requirement)*

### **EPIC 4 - Dashboard planning, designing and development**
- User story (E4US11) - **As a user**, I want to see an overview page that describes the project and dataset so that I understand its purpose. *(Dashboard Requirement)*
- User story (E4US12) - **As a user**, I want to see which variables have the most impact on the sale price so that I can gain insights. *(Business Requirement 1, Dashboard Requirement)*
- User story (E4US13) - **As a user**, I want to input house data and get a real-time prediction of the sale price. *(Business Requirement 2, Dashboard Requirement)*
- User story (E4US14) - **As a user**, I want to see a technical page that shows the model's performance and pipeline so that I can understand how the model works. *(Business Requirement 2, Dashboard Requirement)*

### **EPIC 5 - Documentation and deployment**
- User story (E5US15) - **As a project reviewer**, I want to read a README.md file that describes the project so that I can understand its purpose.


## Technical Implementation of Business Requirements
To address the client's business requirements, we have implemented specific data analysis and machine learning solutions. This section outlines the technical approach used to meet each requirement.

1. **Business Requirement 1**: Identify how house attributes correlate with the sale price.
	- **Rationale**: To understand which factors most influence the sale price, a detailed correlation analysis is required. This helps the client make informed decisions about which attributes are most valuable.
	- **Mapping**:
		- Perform correlation analysis between house attributes and sale price.
		- Visualize the correlations using scatter plots, box plots, and heatmaps.
		- Identify the most correlated variables and present them in an interactive dashboard.
    - **Notebook**: [03 - Correlation Analysis Notebook and Visualization](https://github.com/NatashaRy/house-price-predictor/blob/main/jupyter_notebooks/03-CorrelationStudy.ipynb)
		
2. **Business Requirement 2**: Predict the sale price for the inherited houses and other houses in Ames, Iowa.
	- **Rationale**: By building a machine learning model, we can provide the client with accurate sale price predictions based on house attributes. This helps the client maximize the value of the inherited houses.
	- **Mapping**:
		- Train a regression model to predict sale prices.
		- Implement an interactive feature in the dashboard where users can input house attributes and get real-time predictions.
		- Display the predicted prices for the inherited houses and their total value.
    - **Notebook**: [05 - Model Training and Evaluation Notebook](https://github.com/NatashaRy/house-price-predictor/blob/main/jupyter_notebooks/05-ModelTraining.ipynb)
		
3. **Dashboard Requirements**:
	- **Rationale**: The dashboard serves as a central platform to present insights and predictions in a user-friendly way.
	- **Mapping**:
		- Create an overview page describing the project and dataset.
		- Implement pages for correlation analysis, hypothesis validation, and sale price predictions.
		- Include a technical page showing the model's performance and pipeline.


## Dashboard Design
The dashboard was built using [**Streamlit**](https://streamlit.io/), an open-source Python framework for creating interactive web applications. This project adopted Streamlit's **newer single-file** approach (`app.py`) instead of the traditional `multipage.py` structure. This allowed full use of [**Streamlit Themes**](https://docs.streamlit.io/develop/concepts/configuration/theming) and simplified the app's structure.

### Why We Chose the New Streamlit Approach
1. **Enhanced Design**: Streamlit Themes allow full customization of colors, fonts, and sidebar styling, creating a cohesive and professional look.
2. **Simplified Codebase**: The single-file structure reduced complexity, making the app easier to maintain and extend.
3. **Improved User Experience**: Seamless navigation and polished visuals enhance usability.
4. **Future-Proofing**: Aligns with Streamlit's latest features, ensuring scalability and easier updates.

#### Streamlit Theme Customizations
To enhance the app's visual appeal, we implemented the following customizations in the `config.toml` file:
- **Primary Color**: `#45a348` (green) for a positive and professional tone.
- **Background Colors**: Light and clean (`#fdfdf8` for main background, `#e8e8df` for widgets).
- **Fonts**: `sans-serif` for readability and `monospace` for headings and code blocks.
- **Rounded Corners**: `8px` for a modern, softer look.
- **Custom Chart Colors**: Plotly "Pastel2" and Seaborn "Spectral" palettes for consistency.

#### Benefits of This Approach
By adopting this approach, the app achieves:
- A visually cohesive and user-friendly design.
- A simplified and maintainable codebase.
- Compatibility with Streamlit's latest best practices.

This decision allowed us to prioritize both functionality and design, delivering a modern and intuitive dashboard that meets the client's needs.

### Sidebar and Page Descriptions 
#### Sidebar
- **File name**: [`app.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/app.py)  
- **Purpose**: The Streamlit Sidebar enhances user experience by providing intuitive navigation and quick access to key features and project details. It allows users to seamlessly switch between pages, explore dynamic content like the Table of Contents (TOC), and access essential links. The sidebar remains visible at all times, ensuring easy access to navigation, quick links, and project information.

The following image demonstrates the different **sidebar stages**:

![Sidebar Stages](docs/readme-imgs/sidebar-stages.png)

**Content**
1. **Expanding Sidebar Arrow**:  
	- Located at the top right of the sidebar, the arrow allows users to fold or unfold the sidebar.
	- When expanded, the arrow becomes visible on hover, providing a seamless way to toggle its visibility **(2: Hover Effect in the sidebar stages image)**.
2. **Adjustable Sidebar Width**:  
	- Users can resize the sidebar by dragging it in or out.  
	- When hovering over the right edge, a green line appears, indicating that the sidebar is resizable.
	- This feature allows users to adjust the sidebar's width to suit their preferences and screen size, enhancing usability **(3: Resizable Sidebar in the sidebar stages image)**. 
3. **Custom Header**: 
	- Displays the app icon and title ("🏠 House Price Predictor") above the navigation links, giving the sidebar a polished and professional appearance.
4. **Navigation**: **(1: Default Stage)**  
	- The sidebar contains five navigation options, each paired with a [Material Icon](#media) and a page title.  
	- These options allow users to switch between different pages of the app. The navigation items include:  
		- **Project Overview**  
			- Icon: [Material Icon "home"](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&selected=Material+Symbols+Outlined:home:FILL@0;wght@0;GRAD@0;opsz@24)  
		- **Correlation Analysis**  
			- Icon: [Material Icon "analytics"](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&selected=Material+Symbols+Outlined:analytics:FILL@0;wght@0;GRAD@0;opsz@24)  
		- **Hypotheses and Validation**  
			- Icon: [Material Icon "lightbulb"](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&selected=Material+Symbols+Outlined:lightbulb:FILL@0;wght@0;GRAD@0;opsz@24)  
		- **Sale Price Predictor**  
			- Icon: [Material Icon "money_bag"](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&selected=Material+Symbols+Outlined:money_bag:FILL@0;wght@0;GRAD@0;opsz@24)  
		- **Machine Learning Model**  
			- Icon: [Material Icon "smart_toy"](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&selected=Material+Symbols+Outlined:smart_toy:FILL@0;wght@0;GRAD@0;opsz@24)
5. **Dynamic Table of Contents (TOC)**: 
	- A custom function, `create_toc`, located in [`utils.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/utils.py), dynamically generates a TOC for pages with multiple sections.
	- The TOC is styled for clarity and ease of navigation **(4: Sidebar with TOC in the sidebar stages image)**.
	- Pages with TOC:
		- Correlation Analysis
		- Hypotheses and Validation
		- Machine Learning Model
6. **Quick Links**:
	- Includes links to the GitHub repository and the README file for easy access:
		- [GitHub Repository](https://github.com/NatashaRy/house-price-predictor)
		- [README file](https://github.com/NatashaRy/house-price-predictor/blob/main/README.md)
7. **"About this project" Section**:
	- A dedicated section at the bottom of the sidebar provides information about the project, including:
		- The purpose of the app: *"The app predicts house prices in Ames, Iowa, using machine learning. Users can analyze data and make predictions."*
		- Acknowledgment of the [Code Institute Full Stack Software Developer Course](https://codeinstitute.net/se/full-stack-software-development-diploma/) as the context for this project. 
		- Copyright information: *"© 2025 Natasha Rydell. All rights reserved."*
	- This section is always displayed last in the sidebar, ensuring users can always access key project details.

#### Project Overview
- **File name**: [`summary.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/pages/summary.py)
- **Icon**: 🗒️
- **Purpose**: Provide an overview of the project, including its goals, dataset and business requirements.
- **State Business Requirements**: A clear statement of the business requirements.
	1. Identify the most relevant variables correlated with house sale price.
	2. Predict the sale price of the 4 inherited houses and any other house in Amen, Iowa.

The following image demonstrate the **Project Overview** page:

![Project Overview](docs/readme-imgs/project-overview.png)

**Content**:
1. **Introduction**: Explains what the page is about.
2. **Project Summary**: Summarizes the project and its goals.
3. **Dataset Details**: Link to the Kaggle dataset and quick summary of dataset listing:
	- Number of rows
	- Number of columns
	- Target varible
	- Key variables
	- Dataset preview
4. **Business Requirements**: Both project business requirements are listed with links to pages associated with them. 
5. **Link to README**: Link to the README file on GitHub.
	
	
#### Correlation Analysis
- **File name**: [`analysis.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/pages/analysis.py)
- **Icon**: 📈
- **Purpose**: Address Business Requirement 1 by analyzing and visualizing the correlation between house attributes and sale prices.
- **State Business Requirement 1**: Clearly outline the requirement to identify variables correlated with sale prices.

The following image demonstrate the **Correlations Analysis** page in it's default state:

![Correlation Analysis page](docs/readme-imgs/correlation-analysis.png)

**Content**:
1. **Introduciton**: Explains what the page is about.

The following images demonstrates: 

1. Checked checkbox: **Would you like to inspect the raw dataset**.
2. Expanded expander: **Read more for detailed analysis**.

![Inspect raw dataset & expander for more details on analysis](docs/readme-imgs/correlation-inspect-detailed-analysis.png)

2. *Optional* - **Checkbox to inspect raw dataset (1)**: When checked a scrollable table with the 10 first rows of raw dataset is displayed.
3. **Summary of Analysis**: Summary of what the correlation analysis contributed and conclusions along with information box.
	- **Information Box**: Listing 3 key observations from the correlations analysis.
	- **Expander for more detailed analysis (2)**:

The following images demonstrates the different options for **Heatmaps**:

![Heatmaps](docs/readme-imgs/correlation-analysis-heatmaps.png)

4. **Heatmaps**: Divided into two tabs; one for predefined heatmaps and one for create custom heatmap **(1: Heatmaps)**.
	- **Tab 1 - Checkboxes to Display Predefined heatmaps (1: Heatmaps)**: When checked chosen heatmap is displayed, including:
		- Pearson Correlation Heatmap **(3: Predefined Heatmaps)**
		- Spearman Correlation Heatmap **(3: Predefined Heatmaps)**
		- PPS Matrix Heatmap **(4: Predefined Heatmaps)**
	- **Tab 2 - Create Custom Heatmap**: Users can select their own variables for correlation analysis using the multiple choice dropdown menu **(5: Custom Heatmap)**.

The following images demonstrates:

1. Checked checkbox **Display Distribution of Target Variable**.
2. Checked checkbox **Display all visualizations for key variables**.

![Target variable distrubution & Bivariate analysis](docs/readme-imgs/correlation-analysis-target-bivariate.png)

5. **Checkbox to Display Target Variable Distribution (1: Target Variable Distrubution)**: When checked a histogram with distrubution of `SalePrice` is displayed.
6. **Checkbox to Display Bivariate Analysis (2: Bivariate Analysis)**: When checked several plots are displayed, showing the relationshop between key variables and the target variable.
 

#### Hypotheses and Validation
- **File name**: [`hypotheses.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/pages/hypotheses.py)
- **Icon**: 🔍
- **Purpose**: Present the hypotheses formulated during the project and describe how they were validated.

The following image demonstrate the **Hypotheses and Validation** page in it's default state with hypothesis 1 chosen to be explored:

![Hypotheses and Validation page](docs/readme-imgs/hypotheses.png)

**Content**:
1. **Introduction**: Explains what the page is about.
2. **Hypotheses and Results**: Summary of the results we have reached when we have tested the hypotheses displayed in an information box.
3. **Explore Hypotheses**: Dropdown that allows the user to select one of the four hypothesis to explore further.
4. **Hypothesis**: Each hypothesis is described with:
	- **Rationale**: Explains the reasoning behind the hypothesis.
	- **Validation**: Describes the methods and analyses used to test it.
	- **Plot image**: Visualizes the relationship between variables.
	- **Result**: Summarizes whether the hypothesis was confirmed, partially confirmed, or rejected.
5. **Dropdown to choose one of the hypotheses**:
	1. Larger houses have higher sale price 						*- Validated.*
	2. Houses with higher overall quality have higher sale price 	*- Validated.*
	3. Newer houses have higher sale price 							*- Validated.*
	4. Houses with garages have higher sale price 					*- Validated.*
6. **Returns chosen hypothesis**.

#### Sale Price Predictor
- **File name**: [`price_predictions.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/pages/price_predictions.py)
- **Icon**: 💰
- **Purpose**: Address Business Requirement 2 by providing predictions for the 4 inherited houses and allowing users to predict prices for other houses.
- **State Business Requirement 2**: Clearly outline the requirement to predict house sale prices.

The following image demonstrate the **Sale Price Predictor** page in it's default state with a prediction made with user inputs:

![Sale Price Predictor](docs/readme-imgs/sale-price-predictor.png)

**Content**:
1. **Introduction**: Explains what the page is about.
2. **Reminder of business requirement 2**
3. **Predict the sale price of inherited houses**: This section is divided into two subheadings, each providing tabs for users to choose how they want to view the data:

	The following images demonstrate the charts of:
	1. **Inherited House Data**.
	2. **Predicted Sale Price for Inherited Houses**

	![Inherited Houses Charts](docs/readme-imgs/sale-price-predictor-charts.png)

	1. **Inherited House Data**: Allows users to explore the raw data of the inherited houses.
		- **Tab 1 - Data Table (Overview images of page)**: Displays a full-width table with data filtered using the best features.
		- **Tab 2 - Chart (1: Inherited House Data)**: Displays a full-width chart visualizing the data.
	2. **Predicted Sale Prices for Inherited Houses**: Focuses on the predicted sale prices for the inherited houses.
		- **Tab 1 - Data Table (Overview images of page)**: Displays a full-width table with the predicted sale prices.
		- **Tab 2 - Chart (2: Predicted Sale Price for Inherited Houses)**: Displays a full-width chart visualizing the predicted sale prices.
4. **Sum of total sale price for all inherited houses**.
5. **Prediction of sale price of your own house**: Allows users to input specific house attributes and receive a predicted sale price based on the entered data.
	- **Input Widgets**: Users can provide the following house attributes using interactive widgets:
		- `GarageArea`: Size of the garage in square feet.
		- `GrLivArea`: Above-ground living area in square feet.
		- `KitchenQual`: Kitchen quality, selected from a dropdown menu.
		- `OverallQual`: Overall material and finish quality, selected on a scale from 1 to 10. Has a help box to explain 1=Poor and 10=Excellent quality.
	- **Question Mark Help**: `OverallQual` has each a help mark explaining what the steps are equal to in quality:
		- 1 = Poor
		- 3 = Fair
		- 5 = Typical/Average
		- 7 = Good
		- 10 = Excellent
	- Button labeled *"Predict Sale Price"*, to predict price based on user input.
		- Returns the predicted sale price based on the inputted house data from the user.

#### Machine Learning Model
- **File name**: [`ml_pipeline_predictions.py`](https://github.com/NatashaRy/house-price-predictor/blob/main/pages/ml_pipeline_prediction.py)
- **Icon**: 🤖
- **Purpose**: Provide an overview of the machine learning model's performance and technical implementation.

The following image demonstrate the **Machine Learning Model** page in it's default state:

![Machine Learning Model](docs/readme-imgs/machine-learning-model.png)

**Content**:
1. **Introduction**: Provides an overview of the page, explaining its purpose to present the machine learning model, its performance, and key insights.
2. **Information box**: Summarizes the model training and evaluation process, including key metrics and highlights of the pipeline's performance.
3. **Machine Learning Pipeline**: Displays the structure of the trained machine learning pipeline, outlining the steps involved in data preprocessing, feature selection, and prediction.
4. **Feature Importance**: 
	- Highlights the most important features used by the model to make predictions.
	- Includes a visualization that shows the relative importance of each feature.
5. **Model Performance**:
	- Evaluates the model's performance on both the training and test datasets.
	- Performance scores are displayed side by side in columns for easy comparison.
	- Metrics include:
		- Mean Absolute Error (MAE)
		- Mean Squared Error (MSE)
		- Root Mean Squared Error (RMSE)
		- R² (R-squared)
7. **Regression Evaluation Plots**:
	- Visualizations that compare actual values and predicted values for both the training and test datasets.
	- Plots are displayed side by side for a clear comparison.

## Plots
### Histogram
The histogram shows the distribution of the sales prices (`SalePrice`) in the dataset. This gives an overview of how the prices are spread out and whether there are any clear peaks or outliers.

![Histogram - Distribution of Sale Price](docs/plots/hist_plot_SalePrice.png)

### Heatmaps
**Pearson Correlation Heatmap**: Visualizes linear relationships between variables and sales price. Higher values (close to 1 or -1) indicate strong relationships.

![Heatmap - Pearson Correlation Heatmap](docs/plots/pearson_correlation_heatmap.png)

**Spearman Correlation Heatmap**: Visualizes strong monotonic relationships.

![Heatmap - Spearman Correlation Heatmap](docs/plots/spearman_correlation_heatmap.png)

**Predictive Power Score (PPS) Heatmap**: Shows the predictive strength between variables and the target variable (`SalePrice`), which is useful for identifying important predictors.

![Heatmap - Predictive Power Score (PPS) Heatmap](docs/plots/pps_heatmap.png)

### Box Plots
Box plots show how different categories (e.g., `KitchenQual` or `OverallQual`) affect the selling price. They help identify differences between groups.

*Price by Excellent Kitchen Quality*

![Box plot - Price by Kitchen Quality](docs/plots/box_plot_price_by_KitchenQual_Ex.png)

*Price by Good Kitchen Quality*

![Box plot - Price by Good Kitchen Quality](docs/plots/box_plot_price_by_KitchenQual_Gd.png)

*Price by Typical Kitchen Quality*

![Box plot - Price by Typical Kitchen Quality](docs/plots/box_plot_price_by_KitchenQual_TA.png)

*Price by Overall Quality*

![Box plot - Price by Overall Quality](docs/plots/box_plot_price_by_OverallQual.png)

### Line Plots
The line charts show trends in sales prices over time, for example based on year built (`YearBuilt`) or year renovated (`YearRemodAdd`).

*Price by Year Built*

![Line plot - Price by Year Built](docs/plots/line_plot_price_by_YearBuilt.png)

*Price by Year Remodeled/Added*

![Line plot - Price by Year Remodeled/Added](docs/plots/line_plot_price_by_YearRemodAdd.png)

### Linear Model Plots
These charts show linear relationships between specific variables (e.g., `GrLivArea`, `GarageArea`) and sales price. They include a regression line to illustrate the trend.

*Price by First Floor Square Feet*

![Linear Model Plots - Price by First Floor Square Feet](docs/plots/lm_plot_price_by_1stFlrSF.png)

*Price by Garage Area*

![Linear Model Plots - Price by Garage Area](docs/plots/lm_plot_price_by_GarageArea.png)

*Price by Garage Year Built*

![Linear Model Plots - Price by Garage Year Built](docs/plots/lm_plot_price_by_GarageYrBlt.png)

*Price by Ground Living Area*

![Linear Model Plots - Price by Ground Living Area](docs/plots/lm_plot_price_by_GrLivArea.png)

*Price by Masonry Vaneer Area*

![Linear Model Plots - Price by Masonry Vaneer Area](docs/plots/lm_plot_price_by_MasVnrArea.png)

*Price by Total Basement Square Feet*

![Linear Model Plots - Price by Total Basement Square Feet](docs/plots/lm_plot_price_by_TotalBsmtSF.png)

### Regression Performance Plot
Regression Performance Plot is a visualization that compares actual sales prices (`SalePrice`) to the predicted prices from the machine learning model.

![Regression Performance Plot](outputs/ml_pipeline/predict_price/v1/regression_performance.png)

## Testing
This section outlines the testing strategies implemented during the development and deployment of the project. A combination of automated and manual testing was performed to ensure the functionality, reliability, and performance of the dashboard and machine learning model.

### Summary
- All tests passed successfully, confirming that the project meets the client's requirements.
- Key bugs were identified and resolved, ensuring a robust and error-free solution.

### Testing Overview
A combination of automated and manual testing was conducted to ensure the functionality, reliability, and performance of the dashboard and machine learning model. All tests passed successfully, confirming that the project meets the client's requirements and provides a robust solution.

| **Test Type** | **Objective** | **Testing Method** | **Result** |
|---------------------------|---------------------------|--------------------------|-----------------------|
| **Code Quality Testing** | Ensure all Python files comply with PEP8 guidelines. | Automated | Passed  |
| **Functional Testing** | Verify the functionality of the dashboard, including navigation and features.| Manual | Passed  |
| **Responsive Testing** | Ensure the dashboard is fully functional and visually appealing on various devices and screen sizes. | Manual/Simulated | Passed |
| **EPICS and User Stories Testing** | Validate that all EPICS and User Stories meet the business requirements. | Manual | Passed |
| **Model Unit Testing** | Ensure the machine learning model trains and predicts without errors. | Automated | Passed |
| **Notebook Testing** | Validate the execution of all steps in the Jupyter Notebooks. | Manual | Passed |


### Code Quality Testing with PEP8
All Python project files were rigorously checked using the CI Python Linter, accessible at [PEP8CI](https://pep8ci.herokuapp.com/). This tool was employed to verify that the code complies with PEP 8 guidelines, promoting uniformity, clarity, and adherence to best practices throughout the project. The automated linting process identified and resolved formatting inconsistencies, ensuring a clean and professional codebase.

**Files Tested**:
```
app.py 
multipage.py
summary.py
analysis.py
hypotheses.py
price_prediction.py
ml_pipeline_prediction.py	
prediction_analysis.py
evaluate_reg.py
data_management.py
```

### Functional Testing
The following tests were performed manually to ensure the functionality of the dashboard:

1. **Dataset Loading**: Verified that the dataset loads correctly without errors.
2. **Correlation Analysis**: Checked that the visualizations display the correct relationships between variables.
3. **Prediction Functionality**: Tested the prediction feature by inputting various house attributes and verifying the output.
4. **Dashboard Navigation**: Ensured that all pages in the dashboard are accessible and display the correct content.

**Result**: All manual tests passed successfully, confirming that the dashboard functions as intended.

### Responsive Testing
To ensure the dashboard is fully functional and visually appealing on various devices and screen sizes, the following tests were conducted:

**Summary of Tests:**
- **Navigation**: Verified that the sidebar is properly aligned, foldable/expandable, and functional across all devices.
- **Widgets**: Ensured that all input widgets (e.g., checkboxes, radio buttons, and prediction widgets) work as expected.
- **Visualizations**: Confirmed that all visualizations are scrollable, readable, and adjust correctly to different screen sizes.
- **Tables**: Verified that tables are scrollable and display data correctly on all devices.

#### **Desktop Testing**:

| **Screen Size** (px) | **Testing Method** | **Result** |
|-----------------------|--------------------|------------|
| 2650x1080            | Physical Device    | Passed     |
| 1920x1080            | Physical Device    | Passed     |

- Verified that the dashboard scales correctly on large screens.

#### **Tablet Testing**:

| **Model**            | **Screen Size** (px) | **Testing Method** | **Result** |
|-----------------------|----------------------|--------------------|------------|
| Apple iPad Air 4     | 1640x2360           | Physical Device    | Passed     |
| Samsung Galaxy A7    | 768x1024            | Physical Device    | Passed     |
| Apple iPad Pro       | 1024x1366           | Chrome DevTools    | Passed     |

- Confirmed that all elements adjust to the smaller screen size without overlapping.

#### **Mobile Testing**:

| **Model**            | **Screen Size** (px) | **Testing Method** | **Result** |
|-----------------------|----------------------|--------------------|------------|
| Samsung Galaxy Z Flip6 | 1080x2640          | Physical Device    | Passed     |
| Samsung Galaxy A54   | 1080x2340           | Physical Device    | Passed     |
| Apple iPhone SE      | 375x667             | Chrome DevTools    | Passed     |

- Checked that visualizations are scrollable and readable on smaller screens.

**Result**: All tests passed successfully. The dashboard is fully responsive, ensuring a seamless user experience across desktop, tablet, and mobile devices.

### EPICS and User Stories Testing
To ensure that all business requirements were met, each EPIC and its associated User Stories were tested thoroughly. Below is a summary of the testing process and results for each EPIC.

#### **EPIC 1 - Information Gathering and Data Collection**
- **User Story (E1US01)**: **As a data analyst**, I want to download the dataset from Kaggle so that I can start analyzing it.
  - **Test**: Verified that the dataset was successfully downloaded and matched the expected structure (e.g., correct number of rows and columns).
  - **Result**: Passed.
- **User Story (E1US02)**: **As a data analyst**, I want to explore the structure of the dataset so that I can identify key variables.
  - **Test**: Conducted exploratory data analysis (EDA) to identify key variables and ensure data integrity.
  - **Result**: Passed.
- **User Story (E1US03)**: **As a data analyst**, I want to document the business requirements so that I can ensure the project meets the client's needs.
  - **Test**: Verified that the documented requirements meets the client's goals and were clearly stated in the README file.
  - **Result**: Passed.

#### **EPIC 2 - Data Visualization, Cleaning, and Preparation**
- **User Story (E2US04)**: **As a data analyst**, I want to clean the dataset by handling missing values and outliers so that it is ready for analysis.
  - **Test**: Verified that missing values were handled correctly (e.g., imputation for `LotFrontage`) and outliers were addressed using Winsorization.
  - **Result**: Passed.
- **User Story (E2US05)**: **As a data analyst**, I want to perform a correlation analysis so that I can identify which variables have the most impact on the sale price.
  - **Test**: Verified that correlation matrices and visualizations (e.g., heatmaps) were generated correctly and provided meaningful insights.
  - **Result**: Passed.
- **User Story (E2US06)**: **As a data scientist**, I want to create visualizations of the correlations so that I can present insights to the user.
  - **Test**: Checked that scatter plots, box plots, and heatmaps were displayed correctly in the dashboard.
  - **Result**: Passed.

#### **EPIC 3 - Model Training, Optimization, and Validation**
- **User Story (E3US07)**: **As a data analyst**, I want to split the dataset into training and testing data so that I can evaluate the model's performance.
  - **Test**: Verified that the dataset was split correctly (e.g., 80% training, 20% testing) and that the splits were reproducible.
  - **Result**: Passed.
- **User Story (E3US08)**: **As a data analyst**, I want to build a regression model so that I can predict house prices.
  - **Test**: Verified that the model was trained successfully and produced predictions without errors.
  - **Result**: Passed.
- **User Story (E3US09)**: **As a data analyst**, I want to optimize the model using hyperparameter tuning so that I can improve its performance.
  - **Test**: Verified that GridSearchCV identified the best hyperparameters and improved model performance.
  - **Result**: Passed.
- **User Story (E3US10)**: **As a data analyst**, I want to evaluate the model using R² so that I can ensure it meets the performance requirements.
  - **Test**: Verified that the model achieved an R² score of at least 0.75 on both the training and test sets.
  - **Result**: Passed.

#### **EPIC 4 - Dashboard Planning, Designing, and Development**
- **User Story (E4US11)**: **As a user**, I want to see an overview page that describes the project and dataset so that I understand its purpose.
  - **Test**: Verified that the overview page displayed the project goals, dataset details, and business requirements.
  - **Result**: Passed.
- **User Story (E4US12)**: **As a user**, I want to see which variables have the most impact on the sale price so that I can gain insights.
  - **Test**: Verified that the correlation analysis page displayed the correct variables and visualizations.
  - **Result**: Passed.
- **User Story (E4US13)**: **As a user**, I want to input house data and get a real-time prediction of the sale price.
  - **Test**: Verified that the prediction page accepted user inputs and returned accurate predictions.
  - **Result**: Passed.
- **User Story (E4US14)**: **As a user**, I want to see a technical page that shows the model's performance and pipeline so that I can understand how the model works.
  - **Test**: Verified that the technical page displayed the model's performance metrics and feature importance.
  - **Result**: Passed.

#### **EPIC 5 - Documentation and Deployment**
- **User Story (E5US15)**: **As a project reviewer**, I want to read a README.md file that describes the project so that I can understand its purpose.
  - **Test**: Verified that the README file included all necessary sections (e.g., introduction, CRISP-DM, business requirements, testing) and was clear and well-structured.
  - **Result**: Passed.

### Model Unit Testing
Basic unit tests were performed for the machine learning model to ensure:
- The model trains without errors.
- Predictions are generated for valid input data.
- The model's performance metrics (R², MAE, MSE) meet the expected thresholds.

**Result**: All unit tests passed successfully, confirming that the model is reliable and meets performance expectations.

### Jupyter Notebook Testing
To ensure that all steps in the Jupyter Notebook function correctly and produce the expected results, the following tests were conducted:
1. **Dataset Loading**: Verified that the dataset loads successfully and matches the expected structure (e.g., correct number of rows and columns).
2. **Data Cleaning**: Confirmed that missing values were handled correctly and outliers were addressed using the Winsorizer.
3. **Feature Engineering**: Ensured that transformations (e.g., log and power transformations) were applied correctly to the specified variables.
4. **Correlation Analysis**: Checked that correlation matrices and visualizations were generated without errors.
5. **Model Training**: Verified that the model was trained successfully using GridSearchCV and that the best hyperparameters were identified.
6. **Model Evaluation**: Confirmed that evaluation metrics (R², MAE, MSE) were calculated correctly and met performance expectations.
7. **Predictions**: Tested that predictions were generated correctly for both test data and user input.
8. **Visualizations**: Ensured that all visualizations (e.g., scatter plots, box plots, line plots) were displayed correctly and provided meaningful insights.
9. **Notebook Execution**: Verified that the entire notebook could be executed from start to finish without any errors.

**Result**: All notebook tests passed successfully, confirming that the analysis and modeling steps are reproducible and error-free.

## Bugs 
This section documents the key bugs identified during the development and after the deployment of the project, along with their resolutions. Addressing these issues ensured a robust and error-free solution.

### Bugs Identified During Development
1. **`GarageFinish` missing values**
	- **Issue**: Errors occurred when processing the `GarageFinish` column due to missing values (NaN) and limitations of the `.csv` format.
	- **Cause**: 
		- `.csv` files do not preserve data types, causing `GarageFinish` to be read as `object` instead of `category`, leading to inefficiencies and errors.
		- Missing values (`NaN`) were inconsistently handled, resulting in unexpected behavior.
	- **Resolution**: The issue was resolved by switching to `.parquet` files, which::
		- Preserve data types, ensuring `GarageFinish` remains as `category`.
		- Handle missing values (`NaN`) consistently.
		- Improve performance with faster read/write operations and smaller file sizes.
	- **Impact**: This resolution improved data processing efficiency and ensured consistent handling of missing values, reducing errors during analysis and modeling.

2. **`FutureWarning` message**
	- **Issue**: Multiple `FutureWarning` messages were triggered during development.
	- **Cause**: These warnings were caused by updates in libraries (e.g., pandas, Seaborn) that deprecated certain functions or changed their behavior, including:
		- Deprecation of `is_categorical_dtype`.
		- Deprecation of `DataFrame.applymap`.
		- Changes in Seaborn's handling of `palette` without `hue`.
	- **Resolution**: To suppress these non-critical warnings and maintain a clean console output, the following code was added:
	```python
	import warnings

	# Ignore FutureWarning
	warnings.filterwarnings("ignore", category=FutureWarning)
	```
	- **Impact**: This resolution ensured a cleaner development environment without affecting the functionality or performance of the model.

3. **ValueError in `price_prediction.py`**
	- **Issue**: A `ValueError` occurred during the prediction process when the user selected `1` (Poor) for `KitchenQual` (Kitchen Quality). The input data (X) contained `NaN` values, which the ExtraTreesRegressor model does not support natively, causing the prediction to fail.
	- **Cause**: The value `1` (Poor) for `KitchenQual` was not adequately represented in the training data. This led to the model being unable to process this input effectively when used in the slider, as it lacked sufficient examples to learn from during training. Consequently, the preprocessing pipeline failed to handle this edge case, resulting in missing values (`NaN`) in the input data.
	- **Resolution**:
		1. Replaced the slider for `KitchenQual` with a dropdown `selectbox` that only includes valid options (`Fair`, T`ypical/Average`, `Good`, and `Excellent`), excluding `Poor` as it was not well-represented in the training data.
		2. Ensured that the dropdown options are mapped to the correct encoded values (`Fa`, `TA`, `Gd`, `Ex`) used by the model.
		3. Updated the preprocessing pipeline to handle any unsupported or missing values by defaulting to `Typical/Average` (`TA`).
		4. Retested the app to confirm the issue was resolved and predictions were accurate.
	
	**Error Message Screenshot**:

	![ValueError price_predictor.py](docs/readme-imgs/valueerror-nan-kitchenqual.png)

	- **Impact**: The error caused the prediction process to fail when users selected `1` (Poor) for `KitchenQual`, preventing the app from displaying predicted house prices. After implementing the dropdown `selectbox` and excluding `Poor` as an option, the app now handles this edge case correctly, and predictions are generated without errors.

### Bugs Identified After Deployment
1. **SyntaxError in `hypotheses.py`**
- **Issue**: A `SyntaxError` was identified in the file `hypotheses.py` on line 64 due to a missing colon (`:`) in an `if` statement.
- **Cause**: The `if` statement was improperly formatted, and there were minor inconsistencies in indentation.
- **Resolution**:
  1. Added the missing colon (`:`) to the `if` statement.
  2. Refactored the code to ensure compliance with PEP8 standards.
  3. Retested the app to confirm the issue was resolved.
- **Impact**: The bug caused the app to crash when navigating to the "Hypotheses and Validation" page. After fixing the issue, the app now functions as expected.

**Error Message Screenshot**:

![SyntaxError hypotheses.py](docs/readme-imgs/syntaxerror-hypoteses.png)

## Future Improvements
While the project meets the client's requirements, there are opportunities to enhance its functionality and scalability. These improvements aim to ensure the model remains relevant, accurate, and user-friendly as new data and technologies become available.

1. **Internal Linking Between Pages**: ***Known limitation***
	- Streamlit currently does not support internal linking between pages within the same tab. This limitation affects navigation between sections on different pages and may reduce user-friendliness.
2. **Model Enhancements**:
   	- Experiment with additional machine learning models (e.g., Gradient Boosting, LightGBM) to further improve prediction accuracy.
   	- Implement feature selection techniques to optimize the model's performance.
3. **Dashboard Features**:
   	- Add more interactive visualizations, such as time-series analysis for trends over the years.
   	- Include a "What-If" analysis tool to allow users to simulate different scenarios.
4. **Data Updates**:
   	- Incorporate more recent housing data to ensure the model remains relevant and accurate.
   	- Add external data sources, such as economic indicators, to enhance predictions.
5. **Scalability**:
   	- Develop an API endpoint for integration with other applications.
   	- Optimize the dashboard for faster performance with larger datasets.

## Deployment
The dashboard was deployed using [Render](https://render.com), a cloud platform that simplifies the deployment of web applications. This ensures that the dashboard is accessible to the client and other users, providing a seamless experience for exploring data and generating predictions.

By using Render, the deployment process was streamlined, allowing for quick updates and reliable hosting of the application.

1. Log in to [Render.com](https://render.com) using GitHub.
2. Click on "New" button > "Web Service".
3. "Source Code" option: 
	1. Below "Git Provider" select "GitHub"
	2. Click on your GitHub username.
	3. Select "Only select repositories" > Select your repository name.
	4. Click "Install" > Verify account 
4. Select repository name at "Source Code!"
5. Choose a unique name for you web service.
6. Select language: `Python 3`.
7. Select branch: main.
8. Select region: Frankfurt (EU Central).
9. Set build command: `pip install -r requirements.txt && ./setup.sh`
10. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, to ensure Streamlit is listening on the correct port.
11. Select instance type: Free (512 mb ram, 0.1 CPU)
12. Set environment variables: `Key: PYTHON_VERSION` `Value: 3.12.6`, follow Render's recommendation and get automatic patch updates
13. Click "Deploy Web Service".

## Technologies and Python Packages
This section provides an overview of the key Python libraries and tools used in the project. Each library was chosen for its specific functionality and contribution to the development of the House Pricing Predictor. Below, we outline the purpose of each library and provide examples of how they were utilized in the project.

### Technologies
- [**GitHub**](https://github.com): A web-based platform for version control and collaboration. It was used to host the project's repository, track changes, and manage branches.
- [**GitHub Codespaces**](https://github.com/features/codespaces): A cloud-based development environment that allows you to write, build, and debug code directly in the browser. It was used to streamline development and ensure consistency across devices.
- [**VScode**](https://code.visualstudio.com/): A lightweight and powerful code editor with support for debugging, syntax highlighting, and extensions. It was used for local development and debugging. *VScode version: 1.103.1 (user setup)*
- [**Streamlit**](https://streamlit.io/): An open-source Python framework for building interactive web applications. It was used to create the dashboard for visualizing predictions, feature importance, and correlation analysis, making the project accessible to users.
- [**Render**](https://render.com/): A cloud platform for deploying web applications. It was used to host the Streamlit dashboard, making it accessible to users.
- [**CI Python Linter**](https://pep8ci.herokuapp.com/): A tool for checking Python code against PEP8 standards. It was used to ensure code quality and consistency throughout the project.
- [**GoFullPage**](https://gofullpage.com/): A browser extension for capturing full-page screenshots. It was used to create responsive design screenshots for the README file.
- [**Adobe Illustrator**](https://www.adobe.com/products/illustrator.html): A vector graphics editor, was used to create and edit image collages for the README file, ensuring a clean and professional presentation.
- [**Photoshop**](https://www.adobe.com/se/products/photoshop.html): A powerful image editing program, was also used to create and edit image collages for the README file. 

### Python Packages
#### **Web Application Framework**
- `Streamlit==1.48.1` - Used to create widgets for user input (e.g., sliders for house attributes) and to display interactive visualizations and prediction results.

#### **Data Manipulation and Analysis**
- `pandas==1.5.3` - Used to load the dataset, clean missing values, and perform exploratory data analysis (EDA). 
- `numpy==1.26.1` - Used for mathematical operations such as calculating correlation coefficients and handling large arrays efficiently.

#### **Data Visualization**
- `matplotlib==3.8.0` - Used to create scatter plots and bar charts for correlation analysis.
- `seaborn==0.13.2` - Used to create heatmaps for visualizing correlations between variables.
- `plotly==5.17.0` - Used to create dynamic visualizations in the dashboard, such as interactive scatter plots.

#### **Machine Learning and Model Training** 
- `scikit-learn==1.3.1` - Used to train regression models, evaluate their performance (e.g., R², MAE, MSE), and preprocess data using `StandardScaler`.
- `xgboost==1.6.2` - Used to train a high-performance regression model for predicting house prices.
- `feature-engine==1.6.1` - Used for encoding categorical variables, imputing missing values, and handling outliers with the `Winsorizer`.
- `imbalanced-learn==0.11.0` - Tools for handling imbalanced datasets.

#### **Data Analysis and Profiling**
- `ppscore==1.3.0` - Used during EDA to identify the most important predictors of house prices.
- `ydata-profiling==4.12.0` (Development only) - Used to create a detailed report of the dataset, including distributions, correlations, and missing values.

#### **Model Persistence and File Operations**
- `joblib` (via scikit-learn) - The trained regression model was saved and loaded using joblib for deployment in the dashboard.
- `os` (Python Standard Library) - Used to dynamically load files and manage paths across different environments.
- `importlib` (Python Standard Library) - Used to load custom modules dynamically in the Streamlit app.

#### **Development and Debugging Tools**
- `warnings`(Python Standard Library) - Warnings related to deprecated features were suppressed during development.
- `re` (Python Standard Library) - Used to clean and preprocess text data in the dataset.

#### **Statistical Analysis**
- `scipy.stats`(via SciPy) - Used to calculate p-values and perform hypothesis testing during the validation process.

#### **Project Structure and Utilities**
- `multipage.py` (Custom module) - Used to organize the dashboard into separate pages for correlation analysis, predictions, and technical details.
- `data_management.py` (Custom module) - Used to load the dataset and cache it for faster access in the dashboard.
- `predictive_analysis.py`(Custom module) - Used to generate predictions for user-inputted house attributes.
- `evaluate_reg.py` (Custom module) - Used to calculate performance metrics such as R² and MAE for the trained models.

## Credits
This project would not have been possible without the support and guidance of the following individuals and resources:

- [**Co-Pilot**](https://code.visualstudio.com/docs/copilot/overview): I frequently used Co-Pilot in VScode to solve problems and double-check spelling and typos, as English is not my native language.
- [**Code Institute - Walkthrough Project 02 - Churnometer**](#https://learn.codeinstitute.net/courses/course-v1:CodeInstitute+DDA101+1/courseware/bba260bd5cc14e998b0d7e9b305d50ec/c83c55ea9f6c4e11969591e1b99c6c35/): Some code in steps [04 - Feature Engineering](https://github.com/NatashaRy/milestone-project-heritage-housing-issues/blob/main/jupyter_notebooks/04-FeatureEngineering.ipynb) and [05 - Model Training and Evaluation](https://github.com/NatashaRy/house-price-predictor/blob/main/jupyter_notebooks/05-ModelTraining.ipynb) was fully or partially adapted from Walkthrough Project 2: Churnometer to progress with my machine learning model.
- [**Abacus AI**](https://abacus.ai/): I've used Abacus AI, a platform created by Bindu Reddy, Arvind Sundararajan, and Siddartha Naidu, throughout the project to discuss and review my code. It was particularly helpful during the data cleaning phase, especially for handling `missing values` in `GarageFinish`, which led me to switch to the `parquet` file format for better compatibility.

### Content
- **Repository Template**: I've used the [repository template](https://github.com/Code-Institute-Solutions/milestone-project-heritage-housing-issues) provided by [Code Institute](https://codeinstitute.net) for this project. 

### Media
- **Screenshot of Streamlit Dashboard on different devices**: Created using [Am I Responsive](https://ui.dev/amiresponsive), a tool to showcase how the dashboard looks on various screen sizes.
- **Emojis used in the dashboard**: [Windows Emojis](https://getemoji.com/) for a playful and user-friendly design.
- **Icons in the navigation menu**: [Google Material Icons](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded) for a clean and modern look.

## Acknowledgements
This project would not have been possible without the support and guidance of the following individuals and resources. Their contributions were invaluable in ensuring the success of this project.

- My mentor, **Mo Shami**, for supporting and encouraging me throughout this project. I couldn't have completed it without him.
- I also want to thank my friends and family for supporting me and giving me the time I've needed to finish this project. 