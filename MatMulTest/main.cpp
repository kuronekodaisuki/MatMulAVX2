#include <iostream>
#include <Windows.h>
#include "MatMul.h"

#define M 160
#define K 320
#define N 480

float A[M * K];
float B[K * N];
float C[M * N];

using namespace MatMul;

int main(int argc, char* argv[])
{
	MATRIX a(M, K, A);
	MATRIX b(K, N, B);
	MATRIX c(M, N, C);

	LARGE_INTEGER start, finish, frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);

	AVX2(a, b, c);
	QueryPerformanceCounter(&finish);
	double usec = (double)(finish.QuadPart - start.QuadPart) * 1000 / frequency.QuadPart;
	printf("%.3f msec %lld\n", usec, frequency.QuadPart);

	return 0;
}
