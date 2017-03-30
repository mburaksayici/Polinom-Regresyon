import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as pf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn import linear_model


data = pd.read_csv("linear.csv")

x = data["metrekare"]
y = data["fiyat"]


x = np.array(x)
y= np.array(y)

a,b,c,d,e,f =  np.polyfit(x,y,5)


z = np.arange(150)

plt.scatter(x,y)

plt.plot(z,a*(z**5)+b*(z**4)+c*(z**3)+d*(z**2)+e*z+f)
plt.show()


