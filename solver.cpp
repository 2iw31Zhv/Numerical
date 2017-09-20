#include "solver.h"
#include "utility.h"

void choleskydcp(int n, 
	double * a)
{
	for (int j = 1; j <= n; ++j)
	{
		for (int k = 1; k <= j - 1; ++k)
		{
			a[plat(j, j)] -= 
			std::pow(a[plat(j, k)], 2.0);
		}

		a[plat(j, j)] = std::sqrt(a[plat(j, j)]);

		for (int i = j + 1; i <= n; ++i)
		{
			for (int k = 1; k <= j - 1; ++k)
			{
				a[plat(i, j)] -= 
				a[plat(i, k)] * a[plat(j, k)];
			}

			a[plat(i, j)] /= a[plat(j, j)];
		}
	}
}

int _spsv_iter_impl(int n,
	double * a,
	double * x_k,
	double * x_k_1,
	double * buffer,
	double * b,
	double omega,
	double delta)
{
	double error = DBL_MAX;

	int times = 0;
	while(error > delta)
	{
		if (++times > MAX_ITERATIONS)
		{
			throw std::runtime_error("iterate too many times!");
		}

		cblas_dcopy(n,
			x_k,
			1,
			buffer,
			1);

		for (int i = 0; i < n; ++i)
		{
			double term1 = x_k[i];

			double * rowHead = a + i * (i + 1) / 2;

			// L part
			double term2 = cblas_ddot(i,
				rowHead,
				1,
				x_k_1,
				1);

			// U part
			for (int j = i + 1; j < n; ++j)
			{
				term2 += a[plat(j + 1, i + 1)] * x_k_1[j];
			}

			double term3 = *(rowHead + i);

			x_k[i] = (1.0 - omega) * term1
			+ omega * (b[i] - term2) / term3;
		}

		error = diffnrmi(n,
			buffer,
			x_k);
	}

	return times;
}

int jacobispsv(int n,
	double * a,
	double * x,
	double * b,
	double delta)
{
	double * y = new double[n];
	int times;
	try
	{
		times = _spsv_iter_impl(n,
			a,
			x,
			y,
			y,
			b,
			1.0,
			delta);
	}
	catch(...)
	{
		delete[] y;
		throw;
	}

	delete[] y;

	return times;
}

int sorspsv(int n,
	double * a,
	double * x,
	double * b,
	double omega,
	double delta)
{
	double * buffer = new double[n];
	int times;
	try
	{
		times = _spsv_iter_impl(n,
			a,
			x,
			x,
			buffer,
			b,
			omega,
			delta);
	}
	catch(...)
	{
		delete[] buffer;
		throw;
	}

	delete[] buffer;
	return times;
}
