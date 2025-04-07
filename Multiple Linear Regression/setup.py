from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df = df.drop('B', axis = 1)
df = df.drop('LSTAT', axis = 1)
y = boston.target

x_2 = df[['RM', 'NOX']]

x_train, x_test, y_train, y_test = train_test_split(x_2, y, random_state=6)

def predict_y(x, b, m1, m2):
    predicted_y = []
    x_2D_array = x.values # converts a 2D dataframe to a 2D array
    for i in range(len(x_2D_array)):
        x1 = x_2D_array[i][0]
        x2 = x_2D_array[i][1]
        y = b + m1 * x1 + m2 * x2
        predicted_y.append(y)
    return predicted_y
