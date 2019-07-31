balance_scale_lin is an NN with 
 - 4 Inputs bounded by [1,5]
 - 4 ReLU hidden units
 - 4 ReLU hidden units
 - 3 linear output neurons 
		x_2_0 (scale leans to the left = l),
		x_2_1 (scale is balanced = b),
		x_2_2 (scale leans to the right = r)

I appended a one-hot layer to the end of the NN.

Then I manually added conditions to find input values, s.t. the scale is balanced 
according to the NN.

A readable version of the encoding can be seen in
	balance_scale_lin_find_b_pretty_print.txt

Human readable gurobi encodings are saved in the .lp files,
while the .mps files contain encodings without numerical loss.

The files
	balance_scale_lin_find_b.txt
	balance_scale_lin_find_only_b.txt
contain console output of gurobi's solution, while 
	balance_scale_lin_find_b_cvc.txt
contains cvc4's output for the fist problem.

The files
	balance_scale_lin_find_b_gurobi_inputs_forward_pass.txt
	balance_scale_lin_find_only_b_gurobi_inputs_forward_pass.txt
contain the variable values in the NN after a forward pass with the 
input values calculated by gurobi.
In the first file, the scale leans to the right as opposed to being balanced.

