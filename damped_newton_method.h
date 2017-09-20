#pragma once
#include "config.h"

typedef double (* func)(double x);

double damped_newton_method(double x0,	//the initial value of the root
	func f, 							//the function f of the equation f(x) == 0
	func df, 							//the gradient of the function f
	double lambda0,						//the initial damping factor
	double delta1, 						//the error |f(x) - 0|
	double delta2);						//the error |x_k - x_k-1|