#include "utility.h"

#include "config.h"

//locate an element for packed storage matrix 
//with uplo = "L" triangular layout
inline
int plat(int i,
	int j)
{
	return (j - 1) + i * (i - 1) / 2;
}

double nrmi(int n,
	const double * x)
{
	CBLAS_INDEX i = cblas_idamax(n,
		x,
		1);

	return std::fabs(x[i]);
}

double diffnrmi(int n,
	const double * x,
	const double * y)
{
	double * temp = new double[n];
	cblas_dcopy(n,
		y,
		1,
		temp,
		1);
	cblas_daxpy(n,
		-1.0,
		x,
		1,
		temp,
		1);
	double diffNrmi = nrmi(n,
		temp);

	delete[] temp;

	return diffNrmi;
}

void hilbertgn(int n,
	double * a)
{
	for (int i = 1; i <= n; ++i)
	{
		int col_iter = i;
		for (int j = 1; j <= i; ++j)
		{
			a[plat(i, j)] = 1.0 / col_iter;
			++col_iter;
		}
	}
}

void _vecgn_impl(int n,
	double * x,
	double value)
{
	for (int i = 0; i < n; ++i)
	{
		x[i] = value;
	}
}

void onegn(int n,
	double * x)
{
	_vecgn_impl(n,
		x,
		1.0);
}

void zerogn(int n,
	double * x)
{
	_vecgn_impl(n,
		x,
		0.0);
}

void maxgn(int n,
	double * x)
{
	_vecgn_impl(n,
		x,
		DBL_MAX);
}

void unign(int n,
	double * x,
	int dim)
{
	zerogn(n, x);
	x[dim] = 1.0;
}
