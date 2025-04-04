#1/13: Introduction to Linear Regression
import codecademylib3_seaborn
import matplotlib.pyplot as plt

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

plt.plot(months, revenue, "o")

plt.title("Sandra's Lemonade Stand")

plt.xlabel("Months")
plt.ylabel("Revenue ($)")

plt.show()

# What do you think the revenue in month 13 would be?
month_13 = 199

#2/13: Points and Lines
import codecademylib3_seaborn
import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

#slope:
m = 12
#intercept:
b = 40
#
y = [m*x + b for x in months]

plt.plot(months, revenue, "o")
plt.plot(months,y)
plt.show()

#3/13: Loss
x = [1, 2, 3]
y = [5, 1, 3]

#y = x
m1 = 1
b1 = 0

#y = 0.5x + 1
m2 = 0.5
b2 = 1

y_predicted1 = [m1*x_value + b1 for x_value in x]
y_predicted2 = [m2*x_value + b2 for x_value in x]
total_loss1 = 0
for i in range(len(y)):
  total_loss1 += (y[i]-y_predicted1[i])**2
total_loss2 = 0
for i in range(len(y)):
  total_loss2 += (y[i]-y_predicted2[i])**2
print(total_loss1)
print(total_loss2)
better_fit = 2

#4/13: Minimizing Loss

#5/13: Gradient Descent for Intercept
def get_gradient_at_b(x, y, m, b):
  diff = 0
  for i in range(0, len(x)):
    y_value = y[i]
    x_value = x[i]
    diff += (y_value - ((m * x_value) + b))
  b_gradient = (-2/len(x))*diff
  return b_gradient

#6/13: Gradient Descent for Slope
def get_gradient_at_b(x, y, m, b):
    diff = 0
    N = len(x)
    for i in range(N):
      y_val = y[i]
      x_val = x[i]
      diff += (y_val - ((m * x_val) + b))
    b_gradient = -2/N * diff
    return b_gradient
  
def get_gradient_at_m(x,y,m,b):
  diff = 0
  U = len(x)
  for i in range(U):
    y_val = y[i]
    x_val = x[i]
    diff += x_val*(y_val - (m*x_val+b))
  m_gradient = diff * (-2/U)
  return m_gradient

#7/13: Put it Together
def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

# Define your step_gradient function here
def step_gradient(x,y, b_current, m_current):
  b_gradient = get_gradient_at_b(x, y, b_current, m_current)
  m_gradient = get_gradient_at_m(x, y, b_current, m_current)
  b = b_current - (0.01 * b_gradient)
  m = m_current - (0.01 * m_gradient)
  return b, m
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

# current intercept guess:
b = 0
# current slope guess:
m = 0

# Call your function here to update b and m
b, m = step_gradient(months, revenue, b, m)
print(b, m)

#8/13: Convergence
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from data import bs, bs_000000001, bs_01

iterations = range(1400)

plt.plot(iterations, bs)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()
num_iterations = 800
convergence_b = 48
