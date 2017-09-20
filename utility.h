#pragma once

//Subroutines:
//------- hilbertgn:
//        Generate a packed Hilbert matrix with row-major layout
//--------------------------------------------------------------
//------- n: The order of the matrix
//------- a: The pointer of the matrix, uplo = 'L'. a_ij is stor
//-ed in a[(j - 1) + i * (i - 1) / 2]
//--------------------------------------------------------------
void hilbertgn(int n,
	double * a);

//------- onegn:
//        Generate a vector where all the elements are one
//--------------------------------------------------------------
//------- n: The size of the vector
//------- x: The pointer of the vector
//--------------------------------------------------------------
void onegn(int n,
	double * x);

//------- zerogn:
//        Generate a vector where all the elements are zero
//--------------------------------------------------------------
//------- n: The size of the vector
//------- x: The pointer of the vector
//--------------------------------------------------------------
void zerogn(int n,
	double * x);

//------- unign:
//        Generate a unit vector along with one defined dim
//--------------------------------------------------------------
//------- n: The size of the vector
//------- x: The pointer of the vector
//------- dim: The dimension that equals to one
//--------------------------------------------------------------
void unign(int n,
	double * x,
	int dim);

//------- maxgn:
//        Generate a vector where all the elements are DBL_MAX
//--------------------------------------------------------------
//------- n: The size of the vector
//------- x: The pointer of the vector
//--------------------------------------------------------------
void maxgn(int n,
	double * x);

//Utilities:
//Get the layout in a packed stored lower matrix
inline
int plat(int i,
	int j);

//Return the infinite norm
double nrmi(int n,
	const double * x);

//Get the infinite norm of the difference between two vectors
double diffnrmi(int n,
	const double * x,
	const double * y);
