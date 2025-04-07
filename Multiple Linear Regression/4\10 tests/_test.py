load_file_in_context('script.py')

module_imported("sklearn.model_selection")
module_imported("pandas")
module_imported("codecademylib3_seaborn")

try:
  y_predict
except NameError:
  fail_tests("Did you remember to define `y_predict`?")

pass_tests()
