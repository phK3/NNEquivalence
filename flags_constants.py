
# controls, if gurobi general constraints are used or own encoding
use_grb_native = False

# controls, whether same bigM is used for a whole ReLU-Inequality system
# or different bigMs for different inequalities in that system
use_asymmetric_bounds = True

# default bound for variables and bigMs
default_bound = 999999

# epsilon used for true greater than, and comparisons that are dependent on order
epsilon = 1e-4

# if this tolerance is exceeded, then two nns are considered not equivalent for radius optimization
not_equiv_tolerance = 0.01

# set this flag to only print bounds for deltas and inputs
# (for smtlib format)
hide_non_deltas = True

# bounds of variables that are meant to be the maximum, extreme or top_k value of a
# set of inputs variables are further tightened
use_context_groups = True

# use true inequalities, for assigning the maximum value or top_k value to a variable,
# s.t. gurobi can't be fooled within it's tolerances.
# For example, if x1 = 1 + 1e-10, x2 = 1 + 2e-10, x3 = 1 - 1e-10 gurobi can't detect the real maximum
# and might choose x3 as the max. But x3 might be the lowest variable in the other NN and thus a spurious
# counterexample would be found for top_1 equivalence, because in reality, for both NNs the max could have
# been x2
use_eps_maximum = True

# Whether to use absolute value in encoding of manhattan distance or directly use (2**(n+1)) inequalities
manhattan_use_absolute_value = True
