#include "types.h"

__global__ void kNN_Kernel(float* input, float* output, const int k, const float r, const float h, const graphic g) {
	int2 index = { (int)(blockDim.x * blockIdx.x + threadIdx.x), (int)(blockDim.y * blockIdx.y + threadIdx.y) };
	float C = 0.0f, knned_input = 0.0f, importance = 0.0f;

	if (index.x < g.w && index.y < g.h) {
		//TODO; Corresponding border of input.
		for (int i = max(index.x - k, 0); i <= min(index.x + k, g.w - 1); i++) {
			for (int j = max(index.y - k, 0); j <= min(index.y + k, g.h - 1); j++) {
				importance = ((index.x - i) * (index.x - i) + (index.y - j) * (index.y - j)) / (r*r) + 
					((input[index.x + index.y * g.w] - input[i + j * g.w]) * (input[index.x + index.y * g.w] - input[i + j * g.w])) / (h*h);
				knned_input += input[i + j * g.w] * expf(-max(importance, 0.0f));
				C += expf(-max(importance, 0.0f));
			}
		}
		knned_input /= C;

		output[index.x + index.y * g.w] = knned_input;
	}
}

void kNNdenoise(float* input, const int k, const float r, const float h, const graphic g) {
	dim3 block3 = { 512, 512, 1 }, thread3 = {g.w / block3.x + 1, g.h / block3.y + 1, 1 };
	float *dev_input, *dev_output;

	cudaError_t cudaStatus;

	// Malloc device array on the gpu kernel.
	cudaStatus = cudaMalloc((void**)& dev_input, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN denoising.\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_output, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN denoising.\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_output, input, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

	kNN_Kernel <<<block3, thread3 >>> (dev_input, dev_output, k, r, h, g);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kNN_Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kNN_Kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(input, dev_output, g.w * g.h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);
}

__device__ float ssd(const int2 p, const int2 q, const float* input, const int w, const graphic g) {
	float toReturn = 0.0f, S;
	//TODO; Corresponding border of input.
	for (int i = max(-w, max(-p.x, -q.x)); i <= min(w, min(g.w - p.x, g.w - q.x)); i++) {
		for (int j = max(-w, max(-p.y, -q.y)); j <= min(w, min(g.h - p.y, g.h - q.y)); j++) {
			toReturn += (input[i + p.x + (p.y + j) * g.w] - input[i + q.x + (q.y + j) * g.w]) * (input[i + p.x + (p.y + j) * g.w] - input[i + q.x + (q.y + j) * g.w]);
		}
	}

	S = (min(w, min(g.w - p.x, g.w - q.x)) + max(-w, max(-p.x, -q.x)) + 1) * (max(-w, max(-p.y, -q.y)) + min(w, min(g.h - p.y, g.h - q.y)) + 1);
	return toReturn / S;
}

__forceinline__ __device__ float norm2(int2 p, int2 q) {
	return (p.x - q.x)* (p.x - q.x) + (p.y - q.y) * (p.y - q.y);
}

__global__ void NLM_Kernel(float* input, float* output, const int k, const int w, const float sigma, const float h, const graphic g) {
	int2 index = { (int)(blockDim.x * blockIdx.x + threadIdx.x), (int)(blockDim.y * blockIdx.y + threadIdx.y) };
	float Z = 0.0f, denoised_input = 0.0f, weight = 0.0f;

	if (index.x < g.w && index.y < g.h) {
		for (int i = max(index.x - k, 0); i <= min(index.x + k, g.w - 1); i++) {
			for (int j = max(index.y - k, 0); j <= min(index.y + k, g.h - 1); j++) {
				weight = expf(-max(ssd(index, int2{ i, j }, input, w, g) / (h * h) + 2 * (sigma * sigma), 0.0f));
				denoised_input += input[i + j * g.w] * weight;
				Z += weight;
			}
		}
		denoised_input /= Z;

		output[index.x + index.y * g.w] = denoised_input;
	}
}

void NLMdenoise(float* input, const int k, const int w, const float sigma, const float h, const graphic g) {
	dim3 block3 = { 512, 512, 1 }, thread3 = { g.w / block3.x + 1, g.h / block3.y + 1, 1 };
	float* dev_input, * dev_output;

	cudaError_t cudaStatus;

	// Malloc device array on the gpu kernel.
	cudaStatus = cudaMalloc((void**)& dev_input, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN denoising.\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_output, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN denoising.\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_output, input, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

	NLM_Kernel <<<block3, thread3 >>> (dev_input, dev_output, k, w, sigma, h, g);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kNN_Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kNN_Kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(input, dev_output, g.w * g.h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);
}



__device__ float Turkey_bi_weight(const float d, const float h) {
	return (0.0f < d && d <= h) ? (1.0f - ((d * d) / (h * h)) * 1.0f - ((d * d) / (h * h))) / 2.0f : 1.0f;
}

__device__ float Wr(const float up, const float uq, const float s) {
	return expf(-max(((up - uq) * (up - uq) / (2 * s * s)), 0.0f));
}

__device__ float Ws(const int2 p, const int2 q, const float r) {
	return expf(-max(norm2(p, q)/(2 * r * r), 0.0f));
}

__global__ void improvedNLM_Kernel(float* input, float* output, const int k, const int w, const float h, const float s, const float r, const graphic g) {
	int2 index = { (int)(blockDim.x * blockIdx.x + threadIdx.x), (int)(blockDim.y * blockIdx.y + threadIdx.y) };
	float Z = 0.0f, denoised_input = 0.0f, weight = 0.0f, dij = 0.0f;

	if (index.x < g.w && index.y < g.h) {
		for (int i = max(index.x - k, 0); i <= min(index.x + k, g.w - 1); i++) {
			for (int j = max(index.y - k, 0); j <= min(index.y + k, g.h - 1); j++) {
				dij = ssd(index, int2{ i, j }, input, w, g);
				weight = expf(-max((dij / (h * h)) + 2 * h * h, 0.0f)) * Turkey_bi_weight(dij, h) * Wr(input[index.x + index.y * g.w], input[i + j * g.w], r) * Ws(index, int2{ i, j }, s);
				denoised_input += input[i + j * g.w] * weight;
				Z += weight;
			}
		}
		denoised_input /= Z;

		output[index.x + index.y * g.w] = denoised_input;
	}
}

void improvedNLMdenoise(float* input, const int k, const int w, const float h, const float s, const float r, const graphic g) {
	dim3 block3 = { 512, 512, 1 }, thread3 = { g.w / block3.x + 1, g.h / block3.y + 1, 1 };
	float* dev_input, * dev_output;

	cudaError_t cudaStatus;

	// Malloc device array on the gpu kernel.
	cudaStatus = cudaMalloc((void**)& dev_input, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN denoising.\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_output, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN denoising.\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_output, input, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

	improvedNLM_Kernel << <block3, thread3 >> > (dev_input, dev_output, k, w, h, s, r, g);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kNN_Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kNN_Kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(input, dev_output, g.w * g.h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at kNN denoising!\n");
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);
}