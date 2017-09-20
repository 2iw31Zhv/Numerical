#include <iostream>
#include <exception>
#include "damped_newton_method.h"


double damped_newton_method(double x0,
	func f, 
	func df, 
	double lambda0, 
	double delta1, 
	double delta2)
{
	int times = 0;
	double xk_1 = DBL_MAX;
	double xk = x0;

	while (std::fabs(f(xk)) > delta1 ||
		   std::fabs(xk - xk_1) > delta2)
	{
		using std::cout;
		using std::endl;
		cout << xk << endl;
		double s = f(xk) / df(xk);
		if (std::fabs(df(xk)) < 1e-6)
		{
			throw std::runtime_error("divided by too small number!");
		}

		double x = xk - s;

		int damp_times = 0;
		double lambda = lambda0;

		while (std::fabs(f(x)) >= std::fabs(f(xk)))
		{
			x = xk - lambda * s;
			lambda /= 2;

			++damp_times;

			if (damp_times > MAX_ITERATIONS)
			{
				throw std::runtime_error("iterate too many times!");
			}
		}

		++times;

		if (times > MAX_ITERATIONS)
		{
			throw std::runtime_error("iterate too many times!");
		}

		xk_1 = xk;
		xk = x;
	}

	return xk;
}

