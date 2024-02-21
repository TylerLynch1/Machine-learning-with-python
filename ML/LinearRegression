import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Your data
data = {
    'SquareFeet': [500, 690, 690, 700, 660, 690, 720, 705, 810, 703, 820, 510],
    'Price': [800, 919, 928, 1092, 949, 938, 1102, 1150, 1230, 1110, 1200, 830]
}

# Creating a DataFrame from the data
df = pd.DataFrame(data)

# Splitting the data into features (X) and target variable (y)
X = df[['SquareFeet']]
y = df['Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Plotting the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('House Prices vs Square Footage')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()

predicted_price = model.predict([[800]])
print("Predicted price for 800 square feet:", predicted_price[0])
