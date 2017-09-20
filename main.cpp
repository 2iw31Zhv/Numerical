#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "defns.h"

using namespace std;

namespace p1
{
	double f1(double x)
	{
		return (std::pow(x, 3) - x - 1);
	}

	double df1(double x)
	{
		return (3 * std::pow(x, 2) - 1);
	}

	double f2(double x)
	{
		return (5 * x - std::pow(x, 3));
	}

	double df2(double x)
	{
		return (5 - 3 * pow(x, 2));
	}

	void test()
	{
		cout << "Problem 2.2 -------------------------\n";
		cout << "Damped Newton Method ----------------\n";

		try
		{
			cout << "x^3 - x - 1 == 0  ---------------\n";
			cout << "From x = 0.6\n";
			double r1 = damped_newton_method(0.6,
				f1,
				df1,
				1.0,
				1e-4,
				1e-4);

			cout << "5x - x^3 == 0  ------------------\n";
			cout << "From x = 1.2\n";
			double r2 = damped_newton_method(1.2,
				f2,
				df2,
				1.0,
				1e-4,
				1e-4);

			std::cout << "The root of f1(x) == 0 is: " << r1 << std::endl;
			std::cout << "The root of f2(x) == 0 is: " << r2 << std::endl;
		}
		catch (std::exception& e)
		{
			std::cout << "Error: " << e.what() << std::endl;
		}
	}
}

namespace p2
{
	void test(int order, double permute)
	{
		cout << "Problem 3.6 -------------------------\n";
		cout << "Hilbert Matrix ----------------------\n";

		cout << "order: " << order << endl;
		cout << "permute: " << permute << endl;

		int numElems = (1 + order) * order / 2;

		double * a = new double[numElems];

		//generate a hilbert matrix
		//---------------------------
		//a_11
		//a_21 a_22
		//a_31 a_32 a_33
		//... ... ... ... ...
		//a_n1 a_n2 a_n3 ... ... a_nn
		hilbertgn(order, a);

		double * x = new double[order];
		onegn(order, x);

		double * y = new double[order];
		zerogn(order, y);

		//calculate y <- A * x
		cblas_dspmv(CblasRowMajor,
			CblasLower,
			order,
			1.0,
			a,
			x,
			1,
			0.0,
			y,
			1);

		//copy b <- y
		double * b = new double[order];
		cblas_dcopy(order,
			y,
			1,
			b,
			1);

		//copy l <- a
		double * l = new double[numElems];
		cblas_dcopy(numElems,
			a,
			1,
			l,
			1);

		choleskydcp(order,
			l);

		//permute
		double * delta_y = new double[order];
		unign(order,
			delta_y,
			0);

		cblas_daxpy(order,
			permute,
			delta_y,
			1,
			y,
			1);

		//y <- (L * L')^-1 y
		//step1---------------------------
		//y <- L^-1 y
		cblas_dtpsv(CblasRowMajor,
			CblasLower,
			CblasNoTrans,
			CblasNonUnit,
			order,
			l,
			y,
			1);

		//step2---------------------------
		//y <- L'^-1 y
		cblas_dtpsv(CblasRowMajor,
			CblasLower,
			CblasTrans,
			CblasNonUnit,
			order,
			l,
			y,
			1);

		//residual: b <- b - A * y
		cblas_dspmv(CblasRowMajor,
			CblasLower,
			order,
			-1.0,
			a,
			y,
			1,
			1.0,
			b,
			1);

		try
		{
			double residual = nrmi(order,
				b);

			std::cout
				<< "The infinite norm of the residual vector is: "
				<< residual << std::endl;

			//error: y <- y - x
			double error = diffnrmi(order,
				y,
				x);

			std::cout
				<< "The infinite norm of the error vector is: "
				<< error << std::endl;
		}
		catch (std::exception& e)
		{
			std::cout << "Error: " << e.what() << std::endl;
			delete[] a;
			delete[] x;
			delete[] y;
			delete[] b;
			delete[] delta_y;
			delete[] l;
			exit(1);
		}

		delete[] a;
		delete[] x;
		delete[] y;
		delete[] b;
		delete[] delta_y;
		delete[] l;
	}
}

namespace p3
{
	void test(double omega, double error)
	{
		int order = 10;

		cout << "Problem 4.1 -------------------------\n";
		cout << "Hilbert Matrix II -------------------\n";

		cout << "omega: " << omega << endl;
		cout << "error: " << error << endl;
		int numElems = (order + 1) * order / 2;
		double * a = new double[numElems];
		hilbertgn(order,
			a);

		double * b = new double[order];

		for (int i = 0; i < order; ++i)
		{
			b[i] = 1.0 / (i + 1);
		}

		double * x = new double[order];
		double * y = new double[order];

		try
		{
			int timesJcb;
			int timesSor;
			timesJcb = jacobispsv(order,
				a,
				x,
				b,
				error);

			timesSor = sorspsv(order,
				a,
				y,
				b,
				omega,
				error);

			std::cout << "The result for Jacobi iteration is: " << std::endl;
			std::cout << "Iterate: " << timesJcb << std::endl;
			for (int i = 0; i < order; ++i)
			{
				std::cout << x[i] << '\t';
			}
			std::cout << std::endl;

			std::cout << "The result for SOR iteration is: " << std::endl;
			std::cout << "Iterate: " << timesSor << std::endl;
			for (int i = 0; i < order; ++i)
			{
				std::cout << y[i] << '\t';
			}
			std::cout << std::endl;
		}
		catch (std::exception& e)
		{
			std::cout << "Error: " << e.what() << std::endl;
			delete[] y;
			delete[] x;
			delete[] b;
			delete[] a;
			exit(1);
		}

		delete[] y;
		delete[] x;
		delete[] b;
		delete[] a;
	}
}

namespace p4
{
	using namespace std;

	void test(string filename)
	{
		cout << "Problem 6.3 -------------------------\n";
		cout << "Fitting I ---------------------------\n";

		ifstream fin(filename);
		double tx, ty;
		vector<double> vecx, vecy;
		while (fin >> tx >> ty)
		{
			vecx.push_back(tx);
			vecy.push_back(ty);
		}

		double * x = &vecx[0];
		double * y = &vecy[0];

		int n = vecx.size();
		double * yp = new double[n];

		cout << "y = a + bt + ct^2 -------\n";
		double a, b, c;
		fitting2(&a, &b, &c, n, x, y);
		cout << "a\tb\tc\n";
		cout << a << '\t' << b << '\t' << c << '\n';

		for (int i = 0; i < n; ++i)
		{
			yp[i] = a + b * x[i] + c * x[i] * x[i];
		}
		cout << "max error: " << diffnrmi(n, yp, y) << endl;

		cout << "y = a * exp(bt)   -------\n";

		fittingexp(&a, &b, n, x, y);
		cout << "a\tb\n";
		cout << a << '\t' << b << '\n';

		for (int i = 0; i < n; ++i)
		{
			yp[i] = a * exp(b * x[i]);
		}
		cout << "max error: " << diffnrmi(n, yp, y) << endl;

		delete[] yp;
	}
}

namespace p5
{
	using namespace std;

	void test(string filename)
	{
		cout << "Problem 6.8 -------------------------\n";
		cout << "Interpolation -----------------------\n";

		ifstream fin(filename);
		if (!fin.is_open())
		{
			cerr << "OPEN FILE ERROR!" << endl;
			exit(1);
		}

		double dy0, dy1;
		fin >> dy0 >> dy1;

		vector<double> xvec, yvec;
		double xtemp, ytemp;
		while (fin >> xtemp >> ytemp)
		{
			xvec.push_back(xtemp);
			yvec.push_back(ytemp);
		}

		double * x = &xvec[0];
		double * y = &yvec[0];

		int n = xvec.size() - 1;

		double * gradient = new double[n];
		gradient1gn(gradient, n, x);

		double * mu = new double[n + 1];
		double * lambda = new double[n + 1];

		_mugn(mu, n, gradient);
		_lambdagn(lambda, n, gradient);

		double * d = new double[n + 1];
		_dgn(d, n, gradient, y, dy0, dy1);

		double * m = new double[n + 1];
		_mgn(m, n, lambda, mu, d);

		cout << setprecision(4);
		cout << fixed;
		cout << "x\ty\td1y\td2y\n";
		auto BSPLINE = [&](double val)
		{
			cout << (int)val << "\t";
			cout << bspline3(n, m, gradient, x, y, val) << "\t";
			cout << bspline3_d1(n, m, gradient, x, y, val) << "\t";
			cout << bspline3_d2(n, m, gradient, x, y, val) << endl;
		};

		BSPLINE(2);
		BSPLINE(30);
		BSPLINE(130);
		BSPLINE(350);
		BSPLINE(515);

		delete[] m;
		delete[] d;
		delete[] lambda;
		delete[] mu;
		delete[] gradient;
	}
}

namespace p6
{
	using namespace std;

	double func1(double x)
	{
		return 1.0 / x;
	}

	double func2(double x)
	{
		return 1.0 / (1.0 + x * x);
	}

	void test()
	{
		cout << "Problem 7.4 -------------------------\n";
		cout << "Integration -------------------------\n";

		cout << setprecision(8);
		cout << fixed;

		cout << "Int 1/x [1, 2]\n";

		cout << simpson_integral(func1, 1, 2, 36) << endl;
		cout << romberg_integral(func1, 1, 2, 5e-9) << endl;
		cout << gauss_integral(func1, 1, 2, 33) << endl;

		cout << setprecision(7);
		cout << fixed;

		cout << "Int 1/(1 + x^2) [0, 1]\n";
		cout << 4 * simpson_integral(func2, 0, 1, 36) << endl;
		cout << 4 * romberg_integral(func2, 0, 1, 5e-9) << endl;
		cout << 4 * gauss_integral(func2, 0, 1, 33) << endl;
	}
}

int main()
{
	while (1)
	{
		cout << "Please enter the order of the problem: ";
		int order;
		cin >> order;

		int matrix_order;
		double permutation;
		double error;
		double omega;
		string filename1("fitting.data");
		string filename2("bspline3.data");

		switch (order)
		{
		case 0:
			return 0;
		case 1:
			p1::test();
			break;
		case 2:
			cout << "Please enter the Hilbert matrix order: ";
			cin >> matrix_order;
			cout << "Please enter the permutation: ";
			cin >> permutation;
			p2::test(matrix_order, permutation);

			break;
		case 3:
			cout << "Please enter omega: ";
			cin >> omega;
			cout << "Please enter error: ";
			cin >> error;
			p3::test(omega, error);

			break;
		case 4:
			p4::test(filename1);
			break;
		case 5:
			p5::test(filename2);
			break;
		case 6:
			p6::test();
			break;
		default:
			break;
		}
		cout << endl;
	}

	return 0;
}