#include "fitting.h"

#include "config.h"
#include "utility.h"
#include "solver.h"

void fitting_impl_(int order,
	double * p,
	int n,
	const double * x,
	const double * y)
{
	double * A = new double[order * n];
	for (int i = 0; i < n; ++i)
	{
		for (int o = 0; o < order; ++o)
		{
			A[order * i + o] = pow(x[i], o);
		}
	}

	double * G = new double[order * (order + 1) / 2];

	for (int i = 1; i <= order; ++i)
	{
		for (int j = 1; j <= i; ++j)
		{
			G[plat(i, j)] = 0.0;
			for (int k = 0; k < n; ++k)
			{
				G[plat(i, j)] += A[k * order + i - 1] * A[k * order + j - 1];
			}
		}
	}

	zerogn(order, p);

	cblas_dgemv(CblasRowMajor,
		CblasTrans,
		n,
		order,
		1.0,
		A,
		order,
		y,
		1,
		0.0,
		p,
		1);

	choleskydcp(order, G);

	cblas_dtpsv(CblasRowMajor,
		CblasLower,
		CblasNoTrans,
		CblasNonUnit,
		order,
		G,
		p,
		1);

	cblas_dtpsv(CblasRowMajor,
		CblasLower,
		CblasTrans,
		CblasNonUnit,
		order,
		G,
		p,
		1);

	delete[] G;
	delete[] A;
}

void fitting2(double * a,
	double * b,
	double * c,
	int n,
	const double * x,
	const double * y)
{
	double * p = new double[3];
	fitting_impl_(3, p, n, x, y);
	*a = p[0];
	*b = p[1];
	*c = p[2];

	delete[] p;
}

void fittingexp(double * a,
	double * b,
	int n,
	const double * x,
	const double * y)
{
	double * logy = new double[n];
	for (int i = 0; i < n; ++i)
	{
		logy[i] = log(y[i]);
	}
	double * p = new double[2];

	fitting_impl_(2, p, n, x, logy);

	*a = exp(p[0]);
	*b = p[1];

	delete[] logy;
}

