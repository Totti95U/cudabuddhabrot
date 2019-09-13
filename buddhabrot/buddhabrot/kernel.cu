
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 1280
#define HEIGHT 720


typedef struct {
	int w;
	int h;
	double ratio;

	double dx;
	double dy;

	double max_real;
	double min_real;
	double max_imag;
	double min_imag;
} graphic;

typedef struct {
	float real;
	float imag;
} complex;

typedef struct {
	int samples_per_thread;
	int min_iteration;
	int max_iteration;
} iterationContorol;

cudaError_t renderImage(int* buddha, graphic graph, iterationContorol iteration);



__global__ void initRNG(const unsigned int seed, curandStateMRG32k3a_t* states) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	curand_init(seed, index, 0, states + index);
}

__device__ void draw_point(int* buddha, complex z, const graphic g) {
	int xnum, ynum;
	if (g.min_real < z.real && z.real < g.max_real &&
		g.min_imag < z.imag && z.imag < g.max_imag) {
		xnum = (z.real - g.min_real) / g.dx;
		ynum = (z.imag - g.min_imag) / g.dy;

		buddha[xnum + ynum * g.w] += 1;
	}
}

__device__ int checkinMainBulb(complex z) {
	float q = (z.real - 1.0 / 4.0) * (z.real - 1.0 / 4.0) + z.imag * z.imag;
	if (q * (q + (z.real - 1.0 / 4.0)) < (z.imag * z.imag) / 4.0) {
		return 1;
	}
	else {
		return 0;
	}
}

__device__ int checkinSecondDisc(complex z) {
	if ((z.real + 1) * (z.real + 1) + z.imag * z.imag < 0.25*0.25) {
		return 1;
	}
	else {
		return 0;
	}
}

__global__ void computeBuddhabrot(int* buddha, const graphic graph, iterationContorol iteration, curandStateMRG32k3a_t* states) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	int sample_point;
	complex c, z, z_tmp, z_start;

	for (int i = 0; i < iteration.samples_per_thread; i++) {
		// Generate sample
		c.real = -3 + 6*curand_uniform(&states[index]);
		c.imag = -3 + 6*curand_uniform(&states[index]);


		// Initialize complex number z and flag sample_point
		z_start.real = 0; z_start.imag = 0;

		z = z_start;
		sample_point = 0;

		if (checkinMainBulb(c) || checkinSecondDisc(c))
			continue;

		// Judge whether a point z is escape.
		for (int j = 0; j < iteration.max_iteration; j++) {
			z_tmp.real = z.real * z.real - z.imag * z.imag + c.real;
			z_tmp.imag = 2 * z.real * z.imag + c.imag;
			z = z_tmp;

			if (z.real * z.real + z.imag * z.imag > 16.0) {
				if (j > iteration.min_iteration) {
					sample_point = 1;
				}
				break;
			}
		}

		// sampling
		if (sample_point) {
			// Initialize complex number z
			z = z_start;

			for (int j = 0; j < iteration.max_iteration; j++) {
				z_tmp.real = (z.real * z.real - z.imag * z.imag) + c.real;
				z_tmp.imag = 2 * z.real * z.imag + c.imag;
				z = z_tmp;

				if (z.real * z.real + z.imag * z.imag > 16.0) {
					break;
				}
				else{
					draw_point(buddha, z, graph);
				}
			}
		}
	}
}



int est_min(int* data, unsigned int n) {
	int length = WIDTH * HEIGHT;
	int toReturn[10] = { data[0] };

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

int est_max(int* data, unsigned int n) {
	int length = WIDTH * HEIGHT;
	int toReturn = data[0];

	for (int i = 1; i < length; i++) {
		if (data[i] > toReturn) {
			toReturn = data[i];
		}
	}
	return toReturn;
}

void saveImage(int* data, graphic g) {
	int tmp, min, max;
	FILE* fp = fopen("../../output.pgm", "wb");

	// Write header.
	fprintf(fp, "P2\n%d %d\n%d\n", g.w, g.h, 0xffff);

	min = est_min(data, 1);
	max = est_max(data, 1);

	// Write pixel.
	for (int i = 0; i < g.h; i++) {
		for (int j = 0; j < g.w; j++) {
			tmp = 0xffff * (data[i * g.w + j] - min) / ((double)max);
			fprintf(fp, "%d ", tmp);
		}
		fprintf(fp, "\n");
	}
	 
	fclose(fp);
}



cudaError renderImage(int* buddha, graphic graph, iterationContorol iteration) {
	curandStateMRG32k3a_t* dev_states;
	int* dev_buddha;

	cudaError_t cudaStatus;
	

	const int blocks = 256*256;
	const int threads = 16;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for a vectors (one output).
	cudaStatus = cudaMalloc((void**)& dev_buddha, WIDTH * HEIGHT * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_states, blocks * threads * sizeof(curandStateMRG32k3a_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_buddha, buddha, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Initialize random generator.
	initRNG <<<blocks, threads>>> (1222, dev_states);
	
	// Compute buddhabrot.
	computeBuddhabrot <<<blocks, threads>>> (dev_buddha, graph, iteration, dev_states);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	//Copy output vectors from GPU buffers to host memory.
	cudaStatus = cudaMemcpy(buddha, dev_buddha, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_buddha);
	cudaFree(dev_states);
	
	return cudaStatus;
}



int main()
{
	complex center;
	center.real = -0.5;
	center.imag = 0.0;

	float size = 2.0;

	graphic g;
	g.w = WIDTH;
	g.h = HEIGHT;
	g.ratio = ((double)WIDTH) / HEIGHT;
	g.dx = size / g.h;
	g.dy = size / g.h;
	g.max_real = center.real + 0.5 * size * g.ratio;
	g.max_imag = center.imag + 0.5 * size;
	g.min_real = center.real - 0.5 * size * g.ratio;
	g.min_imag = center.imag - 0.5 * size;

	iterationContorol iteration;
	iteration.samples_per_thread = 100;
	iteration.min_iteration = 1;
	iteration.max_iteration = 400;

	int* buddha = (int*)malloc(sizeof(int) * WIDTH * HEIGHT);
	if (buddha == NULL) {
		printf("Memory cannot be allocated.");
		free(buddha);
		return 1;
	}
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		buddha[i] = 0;
	}

	// compute and render buddhabrot.
	cudaError_t cudaStatus = renderImage(buddha, g, iteration);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderImage failed!");
		return 1;
	}

	// save image of buddhabrot.
	buddha[1] = 1;
	saveImage(buddha, g);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(buddha);

    return 0;
}
