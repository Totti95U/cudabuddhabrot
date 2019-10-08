#pragma once
typedef struct
{
	float real;
	float imag;
} complex;

typedef struct {
	int w;
	int h;
	double ratio;

	float dx;
	float dy;

	complex center;
	float size;
	float max_real;
	float min_real;
	float max_imag;
	float min_imag;

	float gamma;
	float hologram;

	char output[100];

	float sigma;
} graphic;