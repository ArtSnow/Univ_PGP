#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <cstdlib>

using namespace std;

__global__ void kernel(double b, double q, double* ans, long long n) { //отличное от C++ (__global__)
	long long i, idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока 
	long long offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
	for (i = idx; i < n; i += offset) // Для всех требование - внутри цикла for()
		ans[i] = b*pow(q, i);
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	long long n;
	double b = 1;
	double q = 3;
	cin >> n;
	double* answer = (double*)malloc(sizeof(double) * n); //выделение массива ответа

	double* result;
	cudaMalloc(&result, sizeof(double) * n); //выделение массива на устройстве 
	cudaMemcpy(result, answer, sizeof(double) * n, cudaMemcpyHostToDevice);

	kernel <<<256, 256>>> (b, q, result, n); //отличное от C++ (<<<>>>), стандартная функция
	// Многопоточное
	// 256 блоков и 256 потоков(Thread)


	cudaMemcpy(answer, result, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaFree(result);

	cout.precision(10);
	cout.setf(ios::scientific);
	for (long long i = 0; i < n; i++)
		cout << answer[i] << ' ';
	cout << endl;
	free(answer);
	cin >> n;
	return 0;
}
