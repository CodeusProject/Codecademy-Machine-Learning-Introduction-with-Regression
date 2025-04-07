load_file_in_context("script.py")

try:
  multiple_linear_regression(x_train, y_train, x_test)
except NameError:
  fail_tests("Expected `multiple_linear_regression` to be defined, try again!")

def multiple_linear_regression_solution(x_train, y_train, x_test):
  mlr = LinearRegression()
  mlr.fit(x_train, y_train)
  predicted_y = mlr.predict(x_test)
  return predicted_y

learners_answer = multiple_linear_regression(x_train, y_train, x_test)
expected_answer = multiple_linear_regression_solution(x_train, y_train, x_test)

if learners_answer.tolist() != expected_answer.tolist():
	fail_tests("Your model did not calculate predictions accurately, try again!")

pass_tests()
