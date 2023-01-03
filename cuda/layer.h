#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float learning_rate = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int size_kernel, channels, input_size;
	float *output;
	float *pact;
	float *bias;
	float *weight;
	float *back_out;
	float *back_pact;
	float *back_weight;

	Layer(int size_kernel, int channels, int input_size);

	~Layer();

	void load_image(float *data);
	void clear();
	void backprop_clear();
};


// CUDA kernels for different operations
__device__ float sigmoid(float x);
__global__ void activation_function(float *input, float *output, const int N);
__global__ void calc_error(float *err, float *output, unsigned int label, const int N);
__global__ void grad(float *weights, float *grads, const int N);

// CUDA kernels for forward propagation
__global__ void fwd_pact_conv(float input[28][28], float pact[6][24][24], float weight[6][5][5]);
__global__ void fwd_add_bias_conv(float pact[6][24][24], float bias[6]);
__global__ void fwd_pact_sub(float input[6][24][24], float pact[6][6][6], float weight[1][4][4]);
__global__ void fwd_add_bias_sub(float pact[6][6][6], float bias[1]);
__global__ void fwd_pact_full(float input[6][6][6], float pact[10], float weight[10][6][6][6]);
__global__ void fwd_add_bias_full(float pact[10], float bias[10]);

// CUDA kernels for backpropagation
__global__ void bkwd_weight_full(float back_weight[10][6][6][6], float back_pact[10], float output[6][6][6]);
__global__ void bkwd_bias_full(float bias[10], float back_pact[10]);
__global__ void bkwd_output_sub(float back_out[6][6][6], float weight[10][6][6][6], float back_pact[10]);
__global__ void bkwd_pact_sub(float back_pact[6][6][6], float back_out[6][6][6], float pact[6][6][6]);
__global__ void bkwd_weight_sub(float back_weight[1][4][4], float back_pact[6][6][6], float output[6][24][24]);
__global__ void bkwd_bias_sub(float bias[1], float back_pact[6][6][6]);
__global__ void bkwd_output_conv(float back_out[6][24][24], float weight[1][4][4], float back_pact[6][6][6]);
__global__ void bkwd_pact_conv(float back_pact[6][24][24], float back_out[6][24][24], float pact[6][24][24]);
__global__ void bkwd_weight_conv(float back_weight[6][5][5], float back_pact[6][24][24], float output[28][28]);
__global__ void bkwd_bias_conv(float bias[6], float back_pact[6][24][24]);
