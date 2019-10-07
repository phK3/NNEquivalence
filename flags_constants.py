
# controls, if gurobi general constraints are used
# (right now only for binary multiplication)
# TODO: extend to ReLU, Max, ...
use_grb_native = True

# controls, whether same bigM is used for a whole ReLU-Inequality system
# or different bigMs for different inequalities in that system
use_asymmetric_bounds = False

# default bound for variables and bigMs
default_bound = 999999

# epsilon used for true greater than, and comparisons that are dependent on order
epsilon = 1e-8

