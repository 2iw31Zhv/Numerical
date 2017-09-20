#pragma once
#include "config.h"

//------- choleskydcp:
//        Decompose a matrix A into L * L'
//--------------------------------------------------------------
//------- n: The order of the matrix A
//------- a: The pointer of the matrix A, packed stored, with ro
//-w major order
//--------------------------------------------------------------
void choleskydcp(int n, 
	double * a);

//------- jacobispsv:
//        Solve a linear system using Jacobi iteration
//--------------------------------------------------------------
//------- n: The order of the matrix A
//------- a: The pointer of the matrix A, packed stored, with ro
//-w major order
//------- x: The pointer of the solution
//------- b: The right term b = A * x
//------- delta: The error: |x_k - x_k-1|
//--------------------------------------------------------------
int jacobispsv(int n,
	double * a,
	double * x,
	double * b,
	double delta);

//------- sorspsv:
//        Solve a linear system using SOR iteration
//--------------------------------------------------------------
//------- n: The order of the matrix A
//------- a: The pointer of the matrix A, packed stored, with ro
//-w major order
//------- x: The pointer of the solution
//------- b: The right term b = A * x
//------- omega: The relax factor for the solving
//------- delta: The error: |x_k - x_k-1|
//--------------------------------------------------------------
int sorspsv(int n,
	double * a,
	double * x,
	double * b,
	double omega,
	double delta);