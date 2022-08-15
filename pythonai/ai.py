import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import make_regression

df = pd.read_csv("timetable.csv")
df.plot(x='time', y='difficulty', style='o')  
plt.title('time vs diffculty')  
plt.xlabel('time')  
plt.ylabel('difficulty')  
plt.show()
x = df["difficulty"].values.reshape(-1,1)
y = df["time"].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)
plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()
xin = int(input("Difficulty level (1 to 15): "))
if xin > 15:
    xin = 15
xnew = [[xin]]
ynew = regressor.predict(xnew)
print("%s minutes" % (int(ynew[0])))
