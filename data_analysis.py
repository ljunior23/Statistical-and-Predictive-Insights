import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency

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


    

