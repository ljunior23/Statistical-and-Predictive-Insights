# Statistical Analysis and Predictive Modeling of Restaurant Tipping Behavior

## 📌 Project Overview
This project explores restaurant tipping behavior using the **Seaborn Tips Dataset**.  
It combines **exploratory data analysis (EDA)**, **statistical hypothesis testing**,  
and **predictive modeling** to uncover patterns and relationships that influence tips.

## 🎯 Objectives
- Perform **descriptive statistics** and visualize key variables.
- Test for **differences in tipping** by gender using a **t-test**.
- Examine the **relationship between categorical variables** (smoker status & time of day) using a **Chi-Square test**.
- Build a **linear regression model** to predict tip amounts based on the total bill.
- Interpret results and evaluate model performance.

## 📂 Dataset
- **Source:** [Seaborn Tips Dataset](https://github.com/mwaskom/seaborn-data/blob/master/tips.csv)  
- **Description:** Contains information about restaurant bills, tips, and customer demographics.

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| total_bill    | Total bill amount (USD)                          |
| tip           | Tip amount (USD)                                 |
| sex           | Gender of the bill payer                         |
| smoker        | Whether the party included smokers               |
| day           | Day of the week                                  |
| time          | Lunch or Dinner                                  |
| size          | Number of people in the party                    |

## 🛠️ Methods
### 1. Exploratory Data Analysis (EDA)
- Distribution plots of numerical features
- Correlation heatmaps

### 2. Hypothesis Testing
- **Independent T-Test:** Compare tips given by male vs female customers.
- **Chi-Square Test:** Assess the relationship between smoker status and time of day.

### 3. Predictive Modeling
- **Linear Regression:** Predict tip amounts from the total bill.
- Evaluate model performance using **R-squared**.

## 📊 Key Findings
- Significant differences in tipping behavior were found between male and female customers.
- Smoker status and dining time showed statistical dependence.
- Tip amount is positively correlated with the total bill (linear relationship).

## 📈 Example Visuals
- Histogram of total bill distribution
- Correlation heatmap
- Scatterplot with regression line for total bill vs tip

## 💻 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
