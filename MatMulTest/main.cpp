#include <iostream>
#include <Windows.h>
#include "MatMul.h"

#define M 16
#define K 32
#define N 48

float A[M * K];
float B[K * N];
float C[M * N];

int main(int argc, char* argv[])
{
	MATRIX a(M, K, A);
	MATRIX b(K, N, B);
	MATRIX c(M, N, C);

	LARGE_INTEGER start, finish, frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);

	MatMul(a, b, c);
	QueryPerformanceCounter(&finish);
	double usec = (double)(finish.QuadPart - start.QuadPart) * 1000000 / frequency.QuadPart;
	printf("%.3f usec %lld\n", usec, frequency.QuadPart);

	return 0;
}
