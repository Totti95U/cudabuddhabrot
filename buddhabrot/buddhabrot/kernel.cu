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
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <direct.h>
#include <Windows.h>

#include "types.h"
#include "mycomplex.cu"
#include "denoiseFilter.cu"

// #define WIDTH 1280
// #define HEIGHT 720
#define BLOCKS 4096
#define THREADS 256
#define RTGRIDNUM 2024

#define abs(x) (x > 0) ? x : -x

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
complex RotCenter[2];

curandStateMRG32k3a_t* dev_states;

cudaError_t renderImage(unsigned long long int* buddha);

__device__ complex f(complex z, complex c) {
	// z.real = abs(z.real);
	// z.imag = abs(z.imag);
	return expf(z) + c;
}

__global__ void initRNG(const unsigned int seed, curandStateMRG32k3a_t* states) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// standard initirized
	curand_init(seed, index, 0, &states[index]);

	//fast initirized
	// curand_init((seed << 20) + index, 0, 0, &states[index]);
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

__device__ void rot4d(const float* RotMat, const complex *RotCenter, complex* z, const complex* c) {
	float vect[4] = { z->real, z->imag, c->real, c->imag };

	z->real = 0.0f;
	z->imag = 0.0f;
	// c->real = 0.0f;
	// c->imag = 0.0f;

	for (int i = 0; i < 4; i++) {
		z->real += RotMat[i] * (vect[i] - RotCenter[0].real);
		z->imag += RotMat[i + 4] * (vect[i] - RotCenter[0].imag);
		// c->real += RotMat[i + 8] * (vect[i] - RotCenter[1].real);
		// c->imag += RotMat[i + 12] * (vect[i] - RotCenter[1].imag);
	}
}

__global__ void estImportance(int* importance, const graphic g, const iterationContorol iteration, const float* RotMat, const complex *RotCenter) {
	int indexx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int indexy = (blockIdx.y * blockDim.y) + threadIdx.y;
	complex c, z, rotated_z, rotated_c;

	// Initiarize complex num c , z and int importance.
	if (g.type == 0) {
		c.real = -5.0f + 10.0f * indexx / RTGRIDNUM;
		c.imag = -5.0f + 10.0f * indexy / RTGRIDNUM;
		z.real = 0.0f; z.imag = 0.0f;
	}
	else {
		z.real = -5.0f + 10.0f * indexx / RTGRIDNUM;
		z.imag = -5.0f + 10.0f * indexy / RTGRIDNUM;
		c = g.julia_c;
	}
	
	importance[indexx + indexy * RTGRIDNUM] = 0;

	/*
	if (checkinMainBulb(c) || checkinSecondDisc(c)) {
	 	importance[indexx + indexy * RTGRIDNUM] = 0;
		return;
	}
	*/

	for (int i = 0; i < iteration.max_iteration; i++) {
		z = f(z, c);
		if (z.real * z.real + z.imag * z.imag > 32.0f) {
			return;
		}
		else if (i == iteration.max_iteration - 1) {
			importance[indexx + indexy * RTGRIDNUM] = 0;
			return;
		}
		else if (i >= iteration.min_iteration) {
			rotated_z = z; //rotated_c = c;
			rot4d(RotMat, RotCenter, &rotated_z, &c);
			if (checkinWindow(rotated_z, g))
				importance[indexx + indexy * RTGRIDNUM] = 1;
		}
	}
}

__device__ void draw_point(unsigned long long int* buddha, const complex z, const graphic g, const int n) {
	int xnum, ynum;
	if (checkinWindow(z, g)) {
		xnum = (z.real - g.min_real) / g.dx;
		ynum = g.h - (z.imag - g.min_imag) / g.dy;

		if (0 <= xnum && xnum < g.w && 0 <= ynum && ynum < g.h)
			buddha[xnum + ynum * g.w] += n;
	}
}

__device__ complex curand_withtable(curandStateMRG32k3a_t* state, const complex* randTable, const int length) {
	complex toReturn;
	const int index = blockDim.x * blockIdx.x + threadIdx.x;

	int t_index = curand(&state[index]) % length;
	toReturn = randTable[t_index];
	toReturn.real += (-5.0f + 10.0f * curand_uniform(&state[index])) / RTGRIDNUM;
	toReturn.imag += (-5.0f + 10.0f * curand_uniform(&state[index])) / RTGRIDNUM;
	return toReturn;
}

__global__ void computeBuddhabrot(unsigned long long int* buddha, graphic g, iterationContorol iteration, float* RotMat, complex* RotCenter, curandStateMRG32k3a_t* states, const complex* randTable, const int length) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	int sample_point, power = 1, lambda = 1;
	complex c, c_start, z, z_start, tortoise, rotated_z, rotated_c;

	for (int i = 0; i < iteration.samples_per_thread; i++) {
		// Generate sample
		// Initialize complex number z and flag sample_point
		if (g.type == 0) {
			c = curand_withtable(states, randTable, length);
			if (g.hologram < 1) {
				z_start.real = 0.0f; z_start.imag = 0.0f;
			}
			else {
				z_start.real = -1 + 2 * curand_uniform(&states[index]); z_start.imag = -1 + 2 * curand_uniform(&states[index]);
			}
		}
		else {
			z_start = curand_withtable(states, randTable, length);
			if (g.hologram < 1) {
				c = g.julia_c;
			}
			else {
				c.real = g.julia_c.real - g.hologram + 2 * g.hologram * curand_uniform(&states[index]);
				c.imag = g.julia_c.imag - g.hologram + 2 * g.hologram * curand_uniform(&states[index]);
			}
			
		}

		z = z_start;
		tortoise = z;
		sample_point = 0;

	//	if (checkinMainBulb(c) || checkinSecondDisc(c))
	//		continue;

		// Judge whether a point z is escape.
		for (int j = 0; j < iteration.max_iteration; j++) {
			z = f(z, c);

			if (z.real * z.real + z.imag * z.imag > 32.0f) {
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

				if (z.real * z.real + z.imag * z.imag > 32.0f) {
					break;
				}
				else{
					rotated_z = z; //rotated_c = c;
					rot4d(RotMat, RotCenter, &rotated_z, &c);
					draw_point(buddha, rotated_z, g, 1);
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

template <typename T>
T est_min(T* data, unsigned int n) {
	int length = g.w * g.h;
	T toReturn[100] = { 0xffff };

	for (int i = 1; i < length; i++) {
		for (int j = 0; j < n; j++) {
			if (data[i] < toReturn[j]) {
				for (int k = n-1; k > j; k--)
					toReturn[k] = toReturn[k - 1];
				toReturn[j] = data[i];
				break;
			}
		}
	}
	return toReturn[n - 1];
}

template <typename T>
T est_max(T* data, unsigned int n) {
	int length = g.w * g.h;
	T toReturn[100] = { 0 };

	for (int i = 1; i < length; i++) {
		for (int j = 0; j < n; j++) {
			if (data[i] > toReturn[j]) {
				for (int k = n-1; k > j; k--)
					toReturn[k] = toReturn[k - 1];
				toReturn[j] = data[i];
				break;
			}
		}
	}
	return toReturn[n - 1];
}

void saveImage(unsigned long long int* data) {
	unsigned long long int min, max;
	int tmp, gradiation = 0xffff;
	float* normalised_data;
	FILE* fp = fopen(g.output, "wb");

	normalised_data = (float*)malloc(g.w * g.h * sizeof(float));

	// Write header.
	fprintf(fp, "P5\n%d %d\n%d\n", g.w, g.h, 0xff);

	// normalizing.
	min = est_min(data, 1);
	max = est_max(data, 1);
	for (int i = 0; i < g.w * g.h; i++) {
			normalised_data[i] = powf((float)(data[i] - min) / (max - min), 1.0f / g.gamma);
	}
	
	// Denoising.
	NLMdenoise(normalised_data, 5, 3, g.sigma, g.sigma, g);

	// Write pixel.
	for (int i = 0; i < g.w * g.h; i++) {
		tmp = 0xff * normalised_data[i];
		tmp = (tmp > 0xff) ? 0xff : tmp;
		tmp = (tmp < 0) ? 0x00 : tmp;
		putc(tmp, fp);
	}
	
	free(normalised_data);
	fclose(fp);
}



cudaError renderImage(unsigned long long int* buddha) {
	unsigned long long int* dev_buddha;
	complex* dev_randTable, *dev_RotationCenter;
	float* dev_RotationMatrix;

	dim3 rtblocks = { 256, 256, 1 }, rtthreads = { RTGRIDNUM / rtblocks.x, RTGRIDNUM / rtblocks.y, 1 };
	int* dev_importance;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	// subend_t = clock();
	// printf("Initirizing has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching renderImage!\n", cudaStatus);
		goto Error;
	}
	// subend_t = clock();
	//printf("cudaDeviceSynchronize has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

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

	cudaStatus = cudaMalloc((void**)&dev_RotationCenter, 2 * sizeof(complex));
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

	cudaStatus = cudaMemcpy(dev_RotationCenter, RotCenter, 2 * sizeof(complex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	estImportance <<<rtblocks, rtthreads >>> (dev_importance, g, iteration, dev_RotationMatrix, dev_RotationCenter);
	// subend_t = clock();
	// printf("Esting importance has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "estImportance launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching estImportance!\n", cudaStatus);
		goto Error;
	}
	// subend_t = clock();
	// printf("cudaDeviceSynchronize has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

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
	// subend_t = clock();
	// printf("cudaMemcpy dev_importance to importance has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	int sum = 0, rtindex = 0;
	complex c;

	for (int i = 0; i < RTGRIDNUM; i++) {
		for (int j = 0; j < RTGRIDNUM; j++) {
			if (checkImportance(importance, i, j))
				sum++;
		}
	}
	// subend_t = clock();
	// printf("Computing randTable length has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

	complex* randTable = (complex*)malloc(sizeof(complex) * sum);

	for (int i = 0; i < RTGRIDNUM; i++) {
		for (int j = 0; j < RTGRIDNUM; j++) {
			if (checkImportance(importance, i, j)) {
				c.real = -5.0f + 10.0f * i / RTGRIDNUM;
				c.imag = -5.0f + 10.0f * j / RTGRIDNUM;
				randTable[rtindex] = c;
				rtindex++;
			}
		}
	}
	// subend_t = clock();
	// printf("Makin random table has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

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
	computeBuddhabrot <<<BLOCKS, THREADS>>> (dev_buddha, g, iteration, dev_RotationMatrix, dev_RotationCenter, dev_states, dev_randTable, sum);
	// subend_t = clock();
	// printf("Computing buddhabrot has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

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
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeBuddhabrot!\n", cudaStatus);
		goto Error;
	}
	// subend_t = clock();
	// printf("cudaDeviceSynchronize has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);
	
	//Copy output vectors from GPU buffers to host memory.
	cudaStatus = cudaMemcpy(buddha, dev_buddha, g.w * g.h * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	// subend_t = clock();
	// printf("cudaMencpy from dev_buddha to buddha has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

Error:
	cudaFree(dev_buddha);
	cudaFree(dev_randTable);
	cudaFree(dev_RotationMatrix);
	cudaFree(dev_RotationCenter);

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
			else if (strcmp(argv[i], "-holo") == 0) {
				g.hologram = strtof(argv[++i], NULL);
			}
			else if (strcmp(argv[i], "-g") == 0) {
				g.gamma = strtof(argv[++i], NULL);
			}
			else if (strcmp(argv[i], "-o") == 0) {
				sprintf(g.output, argv[++i]);
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
	g.max_real = g.center.real + 0.5f * g.size * (float)g.ratio;
	g.max_imag = g.center.imag + 0.5f * g.size;
	g.min_real = g.center.real - 0.5f * g.size * (float)g.ratio;
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
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			RotationMatrix[j + 4 * i] = (i == j) ? 1.0f : 0.0f;
		}
	}
	for (int n = 0; n < 6; n++) {
		matrix_product(RotationMatrix, rotation[n].RotMat);
	}
}

int main(int argc, char** argv)
{
	// start_t = clock();
	cudaError cudaStatus;
	char tmpfile[260];

	// Default value.
	g.w = 1280;
	g.h = 720;
	g.center.real = -0.0f;
	g.center.imag = -0.0f;
	RotCenter[0] = g.center;
	g.size = 2.6f;// 0.03125f;
	g.hologram = 0.0f;
	g.gamma = 2.0f;
	sprintf(g.output, "../../output.pmg");

	g.sigma = 0.01f;

	g.type = 1;
	g.julia_c.real = -0.0f;
	g.julia_c.imag = -0.0f;

	iteration.samples_per_thread = 16;
	iteration.min_iteration = 0;
	iteration.max_iteration = 100;

	// Initiarize random generator.
	{
		cudaStatus = cudaMalloc((void**)& dev_states, BLOCKS * THREADS * sizeof(curandStateMRG32k3a_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
			goto ErrorMain;
		}

		initRNG <<< BLOCKS, THREADS >>> (1222, dev_states);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching renderImage!\n", cudaStatus);
			goto ErrorMain;
		}
	}

	int max_iterations[3] = { 10, 50, 100 };

	// Initiarize "/pgms" folder.
	for (int i = 0; i < 1000; i++) {
		sprintf(tmpfile, "../../pgms/output%03d.pgm", i);
		DeleteFile(tmpfile);
	}

	for (int n = 0; n < 3; n++) {
		float angles[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 90.0f, 90.0f };
		// Set rotation parameter.
		for (int i = 0; i < 6; i++) {
			rotation[i].axi1 = rotation_axis[2 * i];
			rotation[i].axi2 = rotation_axis[2 * i + 1];
			rotation[i].angl = -angles[i] / 180;// -3.14159265359f *
		}
		sprintf(g.output, "../../pgms/output%03d.pgm", n);

		iteration.max_iteration = max_iterations[n];

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
			goto ErrorMain;
		}

		// save image of buddhabrot.
		saveImage(buddha);
		// subend_t = clock();
		// printf("Saving image has done. (%.2f)\n", (double)(subend_t - start_t) / CLOCKS_PER_SEC);

		free(buddha);
		printf("%03d th image has just been saved.\n", n);
	}

ErrorMain:

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	cudaFree(dev_states);

    return 0;
}
