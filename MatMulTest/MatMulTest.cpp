// MatMulTest.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include <immintrin.h> // AVX2 Intrinsic
#include <omp.h>
#include <stdint.h>


#define NUM_FP32_AVX2 8

struct MATRIX
{
	uint16_t _rows;
	uint16_t _cols;
	float* _data;

	MATRIX(uint16_t rows, uint16_t cols, float* data) : _rows(rows), _cols(cols), _data(data) {}
};

double MatMul(MATRIX& A, MATRIX& B, MATRIX& C)
{
	__m256i idx;
	// bの縦方向にアクセスするときのオフセット
	for (int i = 0; i < NUM_FP32_AVX2; i++)
	{
		idx.m256i_i32[i] = i * B._cols;
	}
#pragma omp paralell for
	for (uint16_t i = 0; i < A._rows; i++)
	{
		float* pC = C._data + i * C._cols;
		for (uint16_t j = 0; j < B._cols; j++)
		{
			float* pA = A._data + i * A._cols;
			float* pB = B._data + j;
			__m256 a, b, c = { 0, 0, 0, 0, 0, 0, 0, 0 };

			// C(i, j)を計算
			for (uint16_t k = 0; k < A._cols; k += NUM_FP32_AVX2)
			{
				a = _mm256_loadu_ps(pA);
				b = _mm256_i32gather_ps(pB, idx, sizeof(float));
				c = _mm256_fmadd_ps(a, b, c);

				// ポインタ更新
				pA += NUM_FP32_AVX2;
				pB += NUM_FP32_AVX2 * B._cols;
			}
			// 結果をC(i,j)に格納
			pC[j] = c.m256_f32[0] + c.m256_f32[1] + c.m256_f32[2] + c.m256_f32[3] + c.m256_f32[4] + c.m256_f32[5] + c.m256_f32[6] + c.m256_f32[7];
		}
	}
	return 0;
}


