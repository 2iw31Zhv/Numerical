#include "config.h"

void gradient1gn(double * gradient, int n, double * x);

double bspline3(int n, 
	double * m, 
	double * gradient,
	double * x,
	double * y,
	double value);

double bspline3_d1(int n,
	double * m,
	double * gradient,
	double * x,
	double * y,
	double value);

double bspline3_d2(int n,
	double * m,
	double * gradient,
	double * x,
	double * y,
	double value);

void _mugn(double * mu, int n, double * gradient);


void _lambdagn(double * lambda, int n, double * gradient);

void _dgn(double * d,
	int n,
	double * gradient,
	double * y,
	double dy0,
	double dy1);

void _mgn(double * mx,
	int n,
	double * lambda,
	double * mu,
	double * d);

int _find_j(int n,
	double * x,
	double value);