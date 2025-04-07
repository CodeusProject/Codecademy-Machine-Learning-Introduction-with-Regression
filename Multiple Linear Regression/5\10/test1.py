import numpy as np
load_file_in_context('script.py')

try:
  print(rm_price.get_offsets())
  print(nox_price.get_offsets())
except NameError:
  fail_tests("Expected `rm_price` and `nox_price` to be defined, try again!")

x1_y1 = rm_price.get_offsets()
x2_y2 = nox_price.get_offsets()
x1 = [i[0] for i in x1_y1]
y1 = [i[1] for i in x1_y1]
x2 = [i[0] for i in x2_y2]
y2 = [i[1] for i in x2_y2]

if not np.array_equal(x_test.RM, x1):
    fail_tests('rm_price : Looks like you did not pass `# of rooms` as x data to `.scatter()`, try again!')
elif not np.array_equal(predicted_y, y1):
  	fail_tests('rm_price : Looks like you did not pass `predicted_y` as y data to `.scatter()`, try again!')
elif not np.array_equal(x_test.NOX, x2):
  	fail_tests('nox_price : Looks like you did not `Nitrogen Oxide concentration` as x data to `.scatter()`, try again!')
elif not np.array_equal(predicted_y, y2):
  	fail_tests('nox_price : Looks like you did not `predicted_y` as y data to `.scatter()`, try again!')
else:
  pass_tests()
  
  
