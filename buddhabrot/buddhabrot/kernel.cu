/*
To delete "warning C4819"
1. Open property of buddhabrot project.
2. Open [CUDA C/C++]/[Command Line].
3. Write "-Xcompiler -wd4819" in additional options.
*/


#include "cuda.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_functions.h"
#include <math_functions.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef __BREAK_ME__
#include <math_functions.h>
#else
#include <cuda_runtime.h>
#endif

#define WIDTH 1280
#define HEIGHT 720
#define RTGRIDNUM 1024

typedef struct {
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

	double gamma;
} graphic;

typedef struct {
	int samples_per_thread;
	int min_iteration;
	int max_iteration;
} iterationContorol;

typedef struct {
	int axi1;
	int axi2;
	float angl;
	float RotMat[16];
} rotationContorol;

// global variances
clock_t start_t, subend_t;

graphic g;
iterationContorol iteration;
rotationContorol rotation[6];
int rotation_axis[6*2] = { 0, 1, 1, 2, 2, 3, 3, 0, 1, 3, 0, 2 };

float RotationMatrix[16] = { 0 };

cudaError_t renderImage(unsigned long long int* buddha);

__device__ complex f(complex z, complex c) {
	complex toReturn;
	toReturn.real = z.real * z.real - z.imag * z.imag + c.real;
	toReturn.imag = 2 * z.real * z.imag + c.imag;
	return toReturn;
}

__global__ void initRNG(const unsigned int seed, curandStateMRG32k3a_t* states) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// standard initirized
	// curand_init(seed, index, 0, &states[index]);

	//fast initirized
	curand_init((seed << 20) + index, 0, 0, &states[index]);
}

__device__ int checkinWindow(complex z, graphic g) {
	if (g.min_real < z.real && z.real < g.max_real &&
		g.min_imag < z.imag && z.imag < g.max_imag) {
		return 1;
	}
	return 0;
}

__device__ int checkinMainBulb(complex z) {
	float q = (z.real - 1.0f / 4.0f) * (z.real - 1.0f / 4.0f) + z.imag * z.imag;
	if (q * (q + (z.real - 1.0f / 4.0f)) < (z.imag * z.imag) / 4.0f) {
		return 1;
	}
	else {
		return 0;
	}
}

__device__ int checkinSecondDisc(complex z) {
	if ((z.real + 1) * (z.real + 1) + z.imag * z.imag < 0.25f * 0.25f) {
		return 1;
	}
	else {
		return 0;
	}
}

__device__ void rot4d(const float* RotMat, complex* z, const complex* c) {
	float vect[4] = { z->real, z->imag, c->real, c->imag };

	z->real = 0.0f;
	z->imag = 0.0f;
	// c->real = 0.0f;
	// c->imag = 0.0f;

	for (int i = 0; i < 4; i++) {
		z->real += RotMat[i] * vect[i];
		z->imag += RotMat[i + 4] * vect[i];
		// c->real += RotMat[i + 8] * vect[i];
		// c->imag += RotMat[i + 12] * vect[i];
	}
}

__global__ void estImportance(int* importance, const graphic g, const iterationContorol iteration, const float* RotMat) {
	int indexx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int indexy = (blockIdx.y * blockDim.y) + threadIdx.y;
	complex c, z, rotated_z, rotated_c;

	// Initiarize complex num c , z and int importance.
	c.real = -3.2f + 6.4f * indexx / RTGRIDNUM;
	c.imag = -3.2f + 6.4f * indexy / RTGRIDNUM;
	z.real = 0.0f; z.imag = 0.0f;
	importance[indexx + indexy * RTGRIDNUM] = 0;

	if (checkinMainBulb(c) || checkinSecondDisc(c)) {
	 	importance[indexx + indexy * RTGRIDNUM] = 0;
		return;
	}

	for (int i = 0; i < iteration.max_iteration; i++) {
		z = f(z, c);
		if (z.real * z.real + z.imag * z.imag > 10.0f) {
			return;
		}
		else if (i == iteration.max_iteration - 1) {
			importance[indexx + indexy * RTGRIDNUM] = 0;
			return;
		}
		else if (i >= iteration.min_iteration) {
			rotated_z = z; //rotated_c = c;
			for (int i = 0; i < 6; i++) {
				rot4d(RotMat, &rotated_z, &c);
			}
			if (checkinWindow(rotated_z, g))
				importance[indexx + indexy * RTGRIDNUM] = 1;
		}
	}
}

__device__ void draw_point(unsigned long long int* buddha, complex z, graphic g) {
	int xnum, ynum;
	if (checkinWindow(z, g)) {
		xnum = (z.real - g.min_real) / g.dx;
		ynum = g.h - (z.imag - g.min_imag) / g.dy;

		buddha[xnum + ynum * g.w] += 1;
	}
}

__device__ complex curand_withtable(curandStateMRG32k3a_t* state, const complex* randTable, const int length) {
	complex toReturn;
	const int index = blockDim.x * blockIdx.x + threadIdx.x;

	int t_index = curand(&state[index]) % length;
	toReturn = randTable[t_index];
	toReturn.real += (-3.2f + 6.4f * curand_uniform(&state[index])) / RTGRIDNUM;
	toReturn.imag += (-3.2f + 6.4f * curand_uniform(&state[index])) / RTGRIDNUM;
	return toReturn;
}

__global__ void computeBuddhabrot(unsigned long long int* buddha, graphic g, iterationContorol iteration, float* RotMat, curandStateMRG32k3a_t* states, const complex* randTable, const int length) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	int sample_point, power = 1, lambda = 1;
	complex c, z, z_start, tortoise, rotated_z, rotated_c;

	for (int i = 0; i < iteration.samples_per_thread; i++) {
		// Generate sample
		c = curand_withtable(states, randTable, length);

		// Initialize complex number z and flag sample_point
		z_start.real = 0; z_start.imag = 0;

		z = z_start;
		tortoise = z;
		sample_point = 0;

		if (checkinMainBulb(c) || checkinSecondDisc(c))
			continue;

		// Judge whether a point z is escape.
		for (int j = 0; j < iteration.max_iteration; j++) {
			z = f(z, c);

			if (z.real * z.real + z.imag * z.imag > 10.0f) {
				if (j >= iteration.min_iteration) {
					sample_point = 1;
				}
				break;
			}
			else if (tortoise.real == z.real && tortoise.imag == z.imag) {
				break;
			}
			else if (power == lambda + 1) {
				tortoise = z;
				power *= 2;
				lambda = 1;
			}
			lambda++;
		}

		// sampling
		if (sample_point) {
			// Initialize complex number z
			z = z_start;

			for (int j = 0; j < iteration.max_iteration; j++) {
				z = f(z, c);

				if (z.real * z.real + z.imag * z.imag > 10.0f) {
					break;
				}
				else{
					rotated_z = z; //rotated_c = c;
					for (int i = 0; i < 6; i++) {
						rot4d(RotMat, &rotated_z, &c);
					}
					draw_point(buddha, rotated_z, g);
				}
			}
		}
	}
}



int checkImportance(const int* importance, const int i, const int j) {
	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			if (-1 < dx + i && dx + i < RTGRIDNUM && -1 < dy + j && dy + j < RTGRIDNUM && importance[(i + dx) + RTGRIDNUM * (j + dy)]) {
				return 1;
			}
		}
	}
	return 0;
}

unsigned long long int est_min(unsigned long long int* data, unsigned int n) {
	int length = g.w * g.h;
	unsigned long long int toReturn[10] = { data[0] };

	for (int i = 1; i < length; i++) {
		for (int j = 0; j < 10; j++) {
			if (data[i] < toReturn[j]) {
				toReturn[j] = data[i];
				break;
			}
		}
	}
	return toReturn[n];
}

unsigned long long int est_max(unsigned long long int* data, unsigned int n) {
	int length = g.w * g.h;
	unsigned long long int toReturn = data[0];

	for (int i = 1; i < length; i++) {
		if (data[i] > toReturn) {
			toReturn = data[i];
		}
	}
	return toReturn;
}

void saveImage(unsigned long long int* data) {
	unsigned long long int tmp, min, max;
	FILE* fp = fopen("../../output.pgm", "wb");

	// Write header.
	fprintf(fp, "P5\n%d %d\n%d\n", g.w, g.h, 0xff);

	min = est_min(data, 1);
	max = est_max(data, 1);

	// Write pixel.
	for (int i = 0; i < g.h; i++) {
		for (int j = 0; j < g.w; j++) {
			tmp = 0xff * pow((double) (data[i * g.w + j] - min) / max, 1/g.gamma);
			putc(tmp, fp);
		}
	}
	 
	fclose(fp);
}


cudaError renderImage(unsigned long long int* buddha) {
	const int blocks = 256 * 256, threads = 512;
	unsigned long long int* dev_buddha;
	complex* dev_randTable;
	float* dev_RotationMatrix;

	dim3 rtblocks = { 256, 256, 1 }, rtthreads = { RTGRIDNUM / rtblocks.x, RTGRIDNUM / rtblocks.y, 1 };
	int* dev_importance;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	// Initiarize random generator.
	curandStateMRG32k3a_t* dev_states;

	cudaStatus = cudaMalloc((void**)& dev_states, blocks * threads * sizeof(curandStateMRG32k3a_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	initRNG <<<blocks, threads >>> (1222, dev_states);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "initRNG launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	subend_t = clock();
	printf("Initirizing has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching renderImage!\n", cudaStatus);
		goto Error;
	}
	subend_t = clock();
	printf("cudaDeviceSynchronize has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	//Make random table.

	cudaStatus = cudaMalloc((void**)& dev_importance, RTGRIDNUM * RTGRIDNUM * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_RotationMatrix, 16 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_RotationMatrix, RotationMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	estImportance <<<rtblocks, rtthreads >>> (dev_importance, g, iteration, dev_RotationMatrix);
	subend_t = clock();
	printf("Esting importance has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "estImportance launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching renderImage!\n", cudaStatus);
		goto Error;
	}
	subend_t = clock();
	printf("cudaDeviceSynchronize has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	int* importance = (int*)malloc(RTGRIDNUM * RTGRIDNUM * sizeof(int));
	for (int i = 0; i < RTGRIDNUM * RTGRIDNUM; i++) {
		importance[i] = 0;
	}

	cudaStatus = cudaMemcpy(importance, dev_importance, sizeof(int) * RTGRIDNUM * RTGRIDNUM, cudaMemcpyDeviceToHost);
	cudaFree(dev_importance);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	subend_t = clock();
	printf("cudaMemcpy dev_importance to importance has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	int sum = 0, rtindex = 0;
	complex c;

	for (int i = 0; i < RTGRIDNUM; i++) {
		for (int j = 0; j < RTGRIDNUM; j++) {
			if (checkImportance(importance, i, j))
				sum++;
		}
	}
	subend_t = clock();
	printf("Computing randTable length has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	complex* randTable = (complex*)malloc(sizeof(complex) * sum);

	for (int i = 0; i < RTGRIDNUM; i++) {
		for (int j = 0; j < RTGRIDNUM; j++) {
			if (checkImportance(importance, i, j)) {
				c.real = -3.2f + 6.4f * i / RTGRIDNUM;
				c.imag = -3.2f + 6.4f * j / RTGRIDNUM;
				randTable[rtindex] = c;
				rtindex++;
			}
		}
	}
	subend_t = clock();
	printf("Makin random table has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	free(importance);

	// Allocate GPU buffers for a vectors (one output).
	cudaStatus = cudaMalloc((void**)& dev_buddha, g.w * g.h * sizeof(unsigned long long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_randTable, sum * sizeof(complex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_buddha, buddha, g.w * g.h * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_randTable, randTable, sum * sizeof(complex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	// Compute buddhabrot.
	computeBuddhabrot <<<blocks, threads>>> (dev_buddha, g, iteration, dev_RotationMatrix, dev_states, dev_randTable, sum);
	subend_t = clock();
	printf("Computing buddhabrot has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeBuddhabrot launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching renderImage!\n", cudaStatus);
		goto Error;
	}
	subend_t = clock();
	printf("cudaDeviceSynchronize has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);
	
	//Copy output vectors from GPU buffers to host memory.
	cudaStatus = cudaMemcpy(buddha, dev_buddha, g.w * g.h * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	subend_t = clock();
	printf("cudaMencpy from dev_buddha to buddha has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

Error:
	cudaFree(dev_states);
	cudaFree(dev_buddha);
	cudaFree(dev_randTable);

	free(randTable);

	return cudaStatus;
}

void matrix_product(float* M, const float* N) {
	float result[16] = { 0 };

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			for (int n = 0; n < 4; n++) {
				result[j + 4 * i] += M[n + 4 * i] * N[j + 4 * n];
			}
		}
	}

	for (int i = 0; i < 16; i++) {
		M[i] = result[i];
	}
}

void set_param(int argc, char** argv) {
	int tmp;

	// Subsitute to parameters.
	if (argc > 1) {
		for (int i = 1; i < argc;) {
			if (strcmp(argv[i], "-w") == 0) {
				g.w = strtol(argv[i], NULL, 10);
				
			}
			else if (strcmp(argv[i], "-h") == 0) {
				g.h = strtol(argv[++i], NULL, 10);
			}
			else if (strcmp(argv[i], "-c") == 0) {
				g.center.real = strtof(argv[++i], NULL);
				g.center.imag = strtof(argv[++i], NULL);
			}
			else if (strcmp(argv[i], "-s") == 0) {
				g.size = strtof(argv[++i], NULL);
			}
			else if (strcmp(argv[i], "-g") == 0) {
				g.gamma = strtod(argv[++i], NULL);
			}
			else if (strcmp(argv[i], "-max") == 0) {
				iteration.max_iteration = strtol(argv[++i], NULL, 10);
			}
			else if (strcmp(argv[i], "-min") == 0) {
				iteration.min_iteration = strtol(argv[++i], NULL, 10);
			}
			else if (strcmp(argv[i], "-sample") == 0) {
				iteration.samples_per_thread = strtol(argv[++i], NULL, 10);
			}
			else if (strcmp(argv[i], "-r") == 0) {
				tmp = strtol(argv[++i], NULL, 10);
				rotation[tmp].angl = 180 * strtof(argv[++i], NULL) / 3.14159265359f;
			}
			else {
				fprintf(stderr, "Invalid options !");
				exit(1);
			}
			i++;
		}
	}

	// Compute
	g.ratio = ((double)g.w) / g.h;
	g.dx = g.size / g.h;
	g.dy = g.size / g.h;
	g.max_real = g.center.real + 0.5f * g.size * g.ratio;
	g.max_imag = g.center.imag + 0.5f * g.size;
	g.min_real = g.center.real - 0.5f * g.size * g.ratio;
	g.min_imag = g.center.imag - 0.5f * g.size;

	for (int n = 0; n < 6; n++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				rotation[n].RotMat[j + 4 * i] = (i == j) ? 1.0f : 0.0f;
			}
		}
		rotation[n].RotMat[rotation[n].axi1 + 4 * rotation[n].axi1] = cospif(rotation[n].angl);
		rotation[n].RotMat[rotation[n].axi2 + 4 * rotation[n].axi1] = -sinpif(rotation[n].angl);
		rotation[n].RotMat[rotation[n].axi1 + 4 * rotation[n].axi2] = sinpif(rotation[n].angl);
		rotation[n].RotMat[rotation[n].axi2 + 4 * rotation[n].axi2] = cospif(rotation[n].angl);
	}
	for (int i = 0; i < 16; i++) {
			RotationMatrix[i] = rotation[0].RotMat[i];
	}
	for (int n = 1; n < 6; n++) {
		matrix_product(RotationMatrix, rotation[n].RotMat);
	}
}

int main(int argc, char** argv)
{
	start_t = clock();

	// Default value.
	g.w = WIDTH;
	g.h = HEIGHT;
	g.center.real = -0.5f; // -0.15943359375f;
	g.center.imag = 0.0f; // 1.034150390625f;
	g.size = 2.6f;// 0.03125f;
	g.gamma = 1.0;

	iteration.samples_per_thread = 2;
	iteration.min_iteration = 0;
	iteration.max_iteration = 1000;

	float angles[6] = { 45.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for (int i=0; i < 6; i++){
		rotation[i].axi1 = rotation_axis[2 * i];
		printf("rotation axis1: %d\n", rotation_axis[2 * i]);
		rotation[i].axi2 = rotation_axis[2 * i + 1];
		printf("rotation axis2: %d\n", rotation_axis[2 * i + 1]);
		rotation[i].angl = -angles[i] / 180;
		printf("rotation angle: %f\n", rotation[i].angl);
	}

	set_param(argc, argv);

	unsigned long long int* buddha = (unsigned long long int*)malloc(sizeof(unsigned long long int) * g.w * g.h);
	if (buddha == NULL) {
		printf("Memory cannot be allocated.\n");
		free(buddha);
		return 1;
	}
	for (int i = 0; i < g.w * g.h; i++) {
		buddha[i] = 0;
	}

	// compute and render buddhabrot.
	cudaError cudaStatus = renderImage(buddha);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderImage failed!\n");
		return 1;
	}

	// save image of buddhabrot.
	saveImage(buddha);
	subend_t = clock();
	printf("Saving image has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	free(buddha);

    return 0;
}
