import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, chi2_contingency

# Load the Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

#  Contigency Table
contigency_table = pd.crosstab(df['smoker'], df['time'])

# Perform CH-Square Test
chi2, p, dof, excepted = chi2_contingency(contigency_table)
print("Chi_Square Statistics: ",  chi2)
print("P-Value: ", p)

# Interpret Result
alpha = 0.05
if p <= alpha:
    print("Reject the null hypothsesis: Variables are dependent")
else:
    print("FailReject the null hypothesis: Variables are independent")




# Visualize Distributions
# sns.histplot(df["total_bill"], kde=True)
# plt.title("Distribution of Total Bill")
# plt.show()

# Correlation heatmap
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

# # Separate data by gender
# male_tips = df[df['sex'] == 'Male']['tip']
# female_tips = df[df['sex'] == 'Female']['tip']

# # Perform t-test
# t_stats, p_value = ttest_ind(male_tips, female_tips)
# print("T-Statistic: ", t_stats)
# print("P-Value: ", p_value)

# # Interpret result
# alpha = 0.05
# if p_value <= alpha:
#     print("Reject all null hypothesis: Siginificant difference")
# else:
#     print("Fail all null hypothesis: no siginficant difference")

# # Difine variables
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values

# # Fit linear regression
model = LinearRegression()
model.fit(X, y)

# # Output the coefficient
print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared: ", model.score(X, y))

# # Plot regression
# sns.scatterplot(x=df['total_bill'], y=df['tip'], label='data', color='blue')
# plt.plot(df['total_bill'], model.predict(X), color="red", label='Regression Line')
# plt.title("Toatl bill vs Tips")
# plt.xlabel("Total_Bill")
# plt.ylabel("Tips")
# plt.legend()
# plt.show()
    

