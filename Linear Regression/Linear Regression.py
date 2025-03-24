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
