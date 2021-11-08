import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('assignment1_dataset.csv')
# print(data.describe())
X = data['age'] + data['bmi']
Y = data['medical_cost']

plt.scatter(X, Y)
plt.xlabel('AGE + BMI', fontsize=15)
plt.ylabel('Medical Cost', fontsize=15)

plt.show()

epochs = 200
Learning_Rate = 0.0000001
m = 0
c = 0
n = len(X)
for i in range(epochs):
    Y_predicted = m * X + c
    Dm = (-2 / n) * sum((Y - Y_predicted) * X)
    Dc = (-2 / n) * sum(Y - Y_predicted)
    m = m - Learning_Rate * Dm
    c = c - Learning_Rate * Dc

Prediction = m * X + c
error = sum((Y - Prediction) ** 2) / n
age_input = int(input("enter age : "))
bmi_input = float(input("enter bmi : "))
predicted_MC = m*(bmi_input+age_input) + c
print("Your Predicted Medical Cost = ", predicted_MC)

print("MSE = ", error)
