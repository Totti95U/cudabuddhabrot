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
	dim3 thread3 = { 256,  256, 1 }, block3 = { g.w / thread3.x + 1, g.h / thread3.y + 1, 1 };
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
	float toReturn = 0.0f, S = 1.0f;
	//TODO; Corresponding border of input.
	for (int i = max(-w, max(-p.x, -q.x)); i <= min(w, min(g.w - p.x, g.w - q.x)); i++) {
		for (int j = max(-w, max(-p.y, -q.y)); j <= min(w, min(g.h - p.y, g.h - q.y)); j++) {
			if (i + p.x + (p.y + j) * g.w < g.w * g.h && i + q.x + (q.y + j) * g.w < g.w * g.h)
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
	dim3 thread3 = { 32, 32, 1 }, block3 = { g.w / thread3.x + 1, g.h / thread3.y + 1, 1 };
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

	NLM_Kernel <<<block3, thread3>>> (dev_input, dev_output, k, w, sigma, h, g);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "MNL_Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching MNL_Kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(input, dev_output, g.w * g.h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at MNL: denoising!\n");
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
	dim3 thread3 = { 256,  256, 1 }, block3 = { g.w / thread3.x + 1, g.h / thread3.y + 1, 1 };
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

	improvedNLM_Kernel <<<block3, thread3>>> (dev_input, dev_output, k, w, h, s, r, g);

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

__global__ void equalization_kernel(float* in_image, float* out_image, unsigned long long int* histgram, unsigned long long int histmin, int ngradiation, graphic g) {
	uint2 index = { blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y };
	if (index.x < g.w && index.y < g.h)
		out_image[index.x + index.y * g.w] = ngradiation * (histgram[(int)in_image[index.x + index.y * g.w]] - histmin) / (g.w * g.h - histmin);
}

template<typename T>
T median3(T x, T y, T z) {
	if (x < y)
		if (y < z) return y; else if (z < x) return x; else return z; else
		if (z < y) return y; else if (x < z) return x; else return z;
}

void quick_sortparm(float* data, float* parm, int left, int right) {
	float tmp, pivot;
	if (left < right) {
		int i = left, j = right;
		tmp, pivot = median3(data[i], data[(i + (j - 1)) / 2], data[j]);
		while (1) {
			while (data[i] < pivot) i++;
			while (data[j] > pivot) j--;
			if (i >= j) break; {
				tmp = data[i]; data[i] = data[j]; data[j] = tmp;
				tmp = parm[i]; parm[i] = parm[j]; parm[j] = tmp;
			}
			i++; j--;
		}
		quick_sortparm(data, parm, left, i - 1);
		quick_sortparm(data, parm, j + 1, right);
	}
}

// ƒqƒXƒgƒOƒ‰ƒ€‹Ïˆê‰»
void equalization(float* image, graphic g, int ngradiation) {
	dim3 thread3 = { 32,  32, 1 }, block3 = { g.w / thread3.x + 1, g.h / thread3.y + 1, 1 };
	unsigned long long int * histgram, * dev_histgram, histmin;
	float *dev_input_image, *dev_output_image;

	cudaError_t cudaStatus;

	histgram = (unsigned long long int*)malloc((ngradiation+1) * sizeof(unsigned long long int));
	if (histgram == NULL) {
		fprintf(stderr, "malloc failed! at equalization.\n");
		goto Error;
	}
	for (int i = 0; i < ngradiation; i++)
		histgram[i] = 0;

	for (int i = 0; i < g.w * g.h; i++) {
		for (int j = (int)image[i]; j <= ngradiation; j++) {
			histgram[j]++;
		}
	}

	histmin = g.w * g.h;
	for (int i = 1; i < ngradiation; i++) {
		if (histmin > histgram[i] && histgram[i] != 0) {
			histmin = histgram[i];
			break;
		}
	}

	
	// Malloc device array on the gpu kernel.
	cudaStatus = cudaMalloc((void**)&dev_input_image, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at equalization.\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output_image, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at equalization.\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_histgram, (ngradiation + 1) * sizeof(unsigned long long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at equalization.\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input_image, image, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at equalization!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_histgram, histgram, (ngradiation + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at equalization!\n");
		goto Error;
	}

	equalization_kernel <<<block3, thread3 >>> (dev_input_image, dev_output_image, dev_histgram, histmin, ngradiation, g);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "equalization_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kNN_Kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(image, dev_output_image, g.w * g.h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at equalization!\n");
		goto Error;
	}

Error:
	free(histgram);
	cudaFree(dev_input_image);
	cudaFree(dev_output_image);
	cudaFree(dev_histgram);
}


__global__ void kernel_kNNequalize(float* in_image, float* out_image, int ngradiation, int k, graphic g) {
	int3 index = { blockDim.x * blockIdx.x + threadIdx.x,blockDim.y * blockIdx.y + threadIdx.y };
	float cdf = 0.0f, cdfmin = 0.0f, minimum_gray = ngradiation;

	if (index.x < g.w && index.y < g.h) {
		for (int i = max(index.y - k, 0); i < min(index.y + k, g.h - 1); i++) {
			for (int j = max(index.x - k, 0); j < min(index.x + k, g.w - 1); j++) {
				if (in_image[j + i * g.w] <= in_image[index.x + index.y * g.w])
					cdf++;
				if (minimum_gray > in_image[j + i * g.w])
					minimum_gray = in_image[j + i * g.w];
			}
		}
		// cdf /= (2 * k + 1) * (2 * k + 1);

		for (int i = max(index.y - k, 0); i <= min(index.y + k, g.h - 1); i++) {
			for (int j = max(index.x - k, 0); j <= min(index.x + k, g.w - 1); j++) {
				if (in_image[j + i * g.w] == minimum_gray)
					cdfmin++;
			}
		}

		out_image[index.x + index.y * g.w] = ngradiation * (cdf - cdfmin) / ((2 * k + 1) * (2 * k + 1) - cdfmin);
	}
}

void kNN_equalization(float* image, int ngradiation, int k, graphic g) {
	dim3 thread3 = { 32, 32, 1 }, block3 = { g.w / thread3.x + 1, g.h / thread3.y + 1, 1 };
	float *dev_input_image, * dev_output_image;
	
	cudaError_t cudaStatus;

	// Malloc device array on the gpu kernel.
	cudaStatus = cudaMalloc((void**)&dev_input_image, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN_equalization.\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output_image, g.w * g.h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! at kNN_equalization.\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_input_image, image, g.w * g.h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! at kNN_equalizetion.\n");
		goto Error;
	}

	kernel_kNNequalize <<<block3, thread3>>> (dev_input_image, dev_output_image, ngradiation, k, g);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel_kNNequalize launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_kNNequalize!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(image, dev_output_image, g.w * g.h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed at equalization!\n");
		goto Error;
	}

Error:
	cudaFree(dev_input_image);
	cudaFree(dev_output_image);
}