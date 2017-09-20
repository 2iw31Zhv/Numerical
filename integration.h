#pragma once
#include "config.h"

typedef double (* func)(double x);

double simpson_integral(func f, 
	double a,
	double b,
	int n);

double romberg_integral(func f,
	double a,
	double b,
	double error);

double gauss_integral(func f,
	double a,
	double b,
	int n);
