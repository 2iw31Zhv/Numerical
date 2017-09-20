#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "bspline3.h"

void gradient1gn(double * gradient, int n, double * x)
{
	cblas_dcopy(n,
		x + 1,
		1,
		gradient,
		1);
	cblas_daxpy(n,
		-1.0,
		x,
		1,
		gradient,
		1);
}


void _mugn(double * mu, int n, double * gradient)
{
	for (int i = 0; i < n - 1; ++i)
	{
		mu[i + 1] = gradient[i] / (gradient[i] + gradient[i + 1]);
	}
	mu[n] = 1.0;
}


void _lambdagn(double * lambda, int n, double * gradient)
{
	lambda[0] = 1.0;
	for (int i = 0; i < n - 1; ++i)
	{
		lambda[i + 1] = gradient[i + 1] / (gradient[i] + gradient[i + 1]);
	}
}

void _dgn(double * d,
 int n, 
 double * gradient, 
 double * y, 
 double dy0,
 double dy1)
 {
 	d[0] = 6.0 * ((y[1] - y[0]) / gradient[0] - dy0) / gradient[0];
 	for (int i = 1; i < n; ++i)
 	{
 		d[i] = 6.0 * (y[i - 1] / (gradient[i - 1] + gradient[i]) / gradient[i - 1]
 			+ y[i + 1] / (gradient[i - 1] + gradient[i]) / gradient[i]
 			- y[i] / gradient[i - 1] / gradient[i]);
 	}
 	d[n] = 6.0 * (dy1 - (y[n] - y[n - 1]) / gradient[n - 1]) / gradient[n - 1];
 }

void _mgn(double * mx, 
	int n, 
	double * lambda, 
	double * mu, 
	double * d)
 {
 	 double * b = new double[n + 1];
 	 double * m = new double[n + 1];
 	 //_vecgn_impl(n + 1, b, 2.0);

 	 for (int i = 0; i < n + 1; ++i)
 	 {
 	 	b[i] = 2.0;
 	 }

 	 for (int i = 1; i <= n; ++i)
 	 {
 	 	m[i] = mu[i] / b[i - 1];
 	 	b[i] -= m[i] * lambda[i-1];
 	 	d[i] -= m[i] * d[i - 1];
 	 }

 	 mx[n] = d[n] / b[n];
 	 for (int i = n - 1; i >= 0; --i)
 	 {
 	 	mx[i] = (d[i] - lambda[i] * mx[i + 1]) / b[i];
 	 }

 	 delete[] m;
 	 delete[] b;

}

int _find_j(int n,
	double * x,
	double value)
{
	std::vector<double> xvec(x, x + n + 1);
	return (std::upper_bound(xvec.begin(), xvec.end(), value) - xvec.begin() - 1);
}

double bspline3(int n, 
	double * m, 
	double * gradient,
	double * x,
	double * y,
	double value)
{
	int j = _find_j(n, x, value);
	assert(j >= 0 && j < n);

	return (m[j] * pow(x[j+1] - value, 3.0) / 6.0
		 + m[j+1] * pow(value - x[j], 3.0) / 6.0 
		 + (y[j] - m[j] * gradient[j] * gradient[j] / 6.0) * (x[j+1] - value)
		 + (y[j+1] - m[j+1] * gradient[j] * gradient[j] / 6.0) * (value - x[j])) / gradient[j];
}

double bspline3_d1(int n,
	double * m,
	double * gradient,
	double * x,
	double * y,
	double value)
{
	int j = _find_j(n, x, value);
	assert(j >= 0 && j < n);

	return -m[j] * (x[j+1] - value) * (x[j+1] - value) / (2.0 * gradient[j])
		  + m[j+1] * (value - x[j]) * (value - x[j]) / (2.0 * gradient[j])
		  + (y[j+1] - y[j]) / gradient[j]
		  - (m[j+1] - m[j]) * gradient[j] / 6.0;
}

double bspline3_d2(int n,
	double * m,
	double * gradient,
	double * x,
	double * y,
	double value)
{
	int j = _find_j(n, x, value);
	assert(j >= 0 && j < n);

	return (m[j] * (x[j+1] - value)
	     + m[j+1] * (value - x[j])) / gradient[j];
}

