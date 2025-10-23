#pragma once

#include <stdint.h>

namespace MatMul
{
	struct MATRIX
	{
		uint16_t _rows;
		uint16_t _cols;
		float* _data;

		MATRIX(uint16_t rows, uint16_t cols, float* data) : _rows(rows), _cols(cols), _data(data) {}
	};

	double AVX2(const MATRIX& A, const MATRIX& B, MATRIX& C);
	//double Naive(const MATRIX& A, const MATRIX& B, MATRIX& C);
}