#include "integration.h"

double simpson_integral(func f, 
	double a,
	double b,
	int n)
{
	double h = (b - a) / n;

	double sum = 0.0;
	for (int k = 0; k < n; ++k)
	{
		sum += f(a + k * h)
		 + 4 * f(a + k * h + 0.5 * h)
		 + f(a + k * h + h);
	}

	return sum * h / 6.0;
}


double romberg_integral(func f,
	double a,
	double b,
	double error)
{
	double h = b - a;
	int n = 0;

	double * Tn_1 = nullptr;
	double * Tn = nullptr;

	Tn = new double[1];
	Tn[0] = 0.5 * h * (f(a) + f(b));

	do
	{
		if (Tn_1 != nullptr)
		{
			delete[] Tn_1;
		}

		Tn_1 = Tn;
		Tn = new double[n + 2];
		Tn[0] = 0.5 * Tn_1[0];
		
		for (int k = 0; k < (1 << n); ++k)
		{
			Tn[0] += 0.5 * h * f(a + k * h + 0.5 * h);
		}
		
		for (int k = 1; k <= n + 1; ++k)
		{
			Tn[k] = (pow(4.0, k) * Tn[k-1] - Tn_1[k-1]) / (pow(4.0, k) - 1.0);
		}
		
		++n;
		h *= 0.5;
	}
	while(fabs(Tn[n] - Tn_1[n-1]) >= error);

	double result = Tn[n];

	delete[] Tn;
	delete[] Tn_1;
	return result;
}

double gauss_integral(func f,
	double a,
	double b,
	int n)
{
	double h = (b - a) / n;

	double sum = 0.0;

	for (int i = 0; i < n; ++i)
	{
		sum += f(a + h * i + h * 0.5 - sqrt(3) * h / 6.0 )
		     + f(a + h * i + h * 0.5 + sqrt(3) * h / 6.0 );
	}

	return sum * h * 0.5;
}