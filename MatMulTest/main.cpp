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

	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

	double usec = MatMul(a, b, c);
	printf("%.3f usec %lld\n", usec, frequency.QuadPart);

	return 0;
}
