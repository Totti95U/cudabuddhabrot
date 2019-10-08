#include "types.h"
// kernel and host code.

template<typename T>
__device__ __host__
complex imagf(T a) {
	complex tmp;
	tmp.real = 0.0f;
	tmp.imag = a;

	return tmp;
}

__device__ __host__
float absf(complex a) {
	return a.real * a.real + a.imag * a.imag;
}

__device__ __host__
float argf(complex a) {
	return atanf(a.imag / a.real);
}

__device__ __host__
complex conjf(complex a) {
	complex tmp;
	tmp.real = a.real;
	tmp.imag = -a.imag;
	return tmp;
}

__device__ __host__
complex operator + (complex a, complex b) {
	complex tmp;
	tmp.real = a.real + b.real;
	tmp.imag = a.imag + b.imag;

	return tmp;
}

__device__ __host__
complex operator - (complex a, complex b) {
	complex tmp;
	tmp.real = a.real - b.real;
	tmp.imag = a.imag - b.imag;

	return tmp;
}

__device__ __host__
complex operator - (complex a) {
	complex tmp;
	tmp.real = -a.real;
	tmp.imag = -a.imag;

	return tmp;
}

template<typename T>
__device__ __host__
complex operator * (T a, complex b) {
	complex tmp;
	tmp.real = a * b.real;
	tmp.imag = a * b.imag;

	return tmp;
}

template<typename T>
__device__ __host__
complex operator * (complex a, T b) {
	return b * a;
}

__device__ __host__
complex operator * (complex a, complex b) {
	complex tmp;
	tmp.real = a.real * b.real - a.imag * b.imag;
	tmp.imag = a.real * b.imag + a.imag * b.real;

	return tmp;
}

__device__ __host__
complex operator / (complex a, complex b) {
	complex tmp;
	tmp.real = (a.real * b.real + a.imag * b.imag) / (b.real * b.real + b.imag * b.imag);
	tmp.imag = (a.imag * b.real - a.real * b.imag) / (b.real * b.real + b.imag * b.imag);

	return tmp;
}

template<typename T>
__device__ __host__
complex operator / (complex a, T b) {
	complex tmp;
	tmp.real = a.real / b;
	tmp.imag = a.imag / b;

	return tmp;
}

template<typename T>
__device__ __host__
complex operator / (T a, complex b) {
	complex tmp;
	tmp.real = a * (b.real) / (b.real * b.real + b.imag * b.imag);
	tmp.imag = a * (-b.imag) / (b.real * b.real + b.imag * b.imag);

	return tmp;
}

__device__ __host__ __forceinline__
complex expf(complex a) {
	complex tmp;
	tmp.real = expf(a.real) * cosf(a.imag);
	tmp.imag = expf(a.real) * sinf(a.imag);

	return tmp;
}

__device__ __host__ __forceinline__
complex sinf(complex a) {
	complex iz;
	iz.real = -a.imag;
	iz.imag = a.real;

	return (expf(iz) + expf(-iz)) / 2;
}

__device__ __host__ __forceinline__
complex cosf(complex a) {
	complex iz;
	iz = a * imagf(1.0f);

	return (expf(iz) - expf(-iz)) / imagf(2.0f);
}

__device__ __host__ __forceinline__
complex tanf(complex a) {
	return sinf(a) / cosf(a);
}

__device__ __host__ __forceinline__
complex sinhf(complex a) {
	return (expf(a) + expf(-a)) / 2;
}

__device__ __host__ __forceinline__
complex coshf(complex a) {
	return (expf(a) - expf(-a)) / 2;
}

__device__ __host__ __forceinline__
complex tanhf(complex a) {
	return sinhf(a)/coshf(a);
}

__device__ __host__ __forceinline__
complex Logf(complex a) {
	complex tmp;
	tmp.real = logf(absf(a));
	tmp.imag = argf(a);

	return tmp;
}
