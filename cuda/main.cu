#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <iostream>
#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_count, test_count;

// Define layers of CNN
static Layer layer_input = Layer(0, 0, 28 * 28);
static Layer layer_conv = Layer(5 * 5, 6, 24 * 24 * 6);
static Layer layer_sub = Layer(4 * 4, 1, 6 * 6 * 6);
static Layer layer_full = Layer(6 * 6 * 6, 10, 10);

// Loading the mnist dataset
static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_count);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_count);
}


// Forward propagation for a single image in the data set
static float forward_pass(double img[28][28])
{
	float input[28][28];

	// read img into input matrix
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = img[i][j];
		}
	}

	// reset to zero before doing forward pass
	layer_input.clear();
	layer_conv.clear();
	layer_sub.clear();
	layer_full.clear();

	// for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// load input image into first layer
	layer_input.load_image((float *)input);
	
	// forward pass for first layer
	fwd_pact_conv<<<64, 64>>>((float (*)[28])layer_input.output, (float (*)[24][24])layer_conv.pact, (float (*)[5][5])layer_conv.weight);
	fwd_add_bias_conv<<<64, 64>>>((float (*)[24][24])layer_conv.pact, layer_conv.bias);
	activation_function<<<64, 64>>>(layer_conv.pact, layer_conv.output, layer_conv.input_size);

	//forward pass for second layer
	fwd_pact_sub<<<64, 64>>>((float (*)[24][24])layer_conv.output, (float (*)[6][6])layer_sub.pact, (float (*)[4][4])layer_sub.weight);
	fwd_add_bias_sub<<<64, 64>>>((float (*)[6][6])layer_sub.pact, layer_sub.bias);
	activation_function<<<64, 64>>>(layer_sub.pact, layer_sub.output, layer_sub.input_size);

	// forward pass for output layer
	fwd_pact_full<<<64, 64>>>((float (*)[6][6])layer_sub.output, layer_full.pact, (float (*)[6][6][6])layer_full.weight);
	fwd_add_bias_full<<<64, 64>>>(layer_full.pact, layer_full.bias);
	activation_function<<<64, 64>>>(layer_full.pact, layer_full.output, layer_full.input_size);
	
	// for timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return ms;
}

// Backward propagation to update the weights
static float back_pass()
{
	// for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// back prop for output layer
	bkwd_weight_full<<<64, 64>>>((float (*)[6][6][6])layer_full.back_weight, layer_full.back_pact, (float (*)[6][6])layer_sub.output);
	bkwd_bias_full<<<64, 64>>>(layer_full.bias, layer_full.back_pact);

	// back prop for second layer
	bkwd_output_sub<<<64, 64>>>((float (*)[6][6])layer_sub.back_out, (float (*)[6][6][6])layer_full.weight, layer_full.back_pact);
	bkwd_pact_sub<<<64, 64>>>((float (*)[6][6])layer_sub.back_pact, (float (*)[6][6])layer_sub.back_out, (float (*)[6][6])layer_sub.pact);
	bkwd_weight_sub<<<64, 64>>>((float (*)[4][4])layer_sub.back_weight, (float (*)[6][6])layer_sub.back_pact, (float (*)[24][24])layer_conv.output);
	bkwd_bias_sub<<<64, 64>>>(layer_sub.bias, (float (*)[6][6])layer_sub.back_pact);

	// back prop for first layer
	bkwd_output_conv<<<64, 64>>>((float (*)[24][24])layer_conv.back_out, (float (*)[4][4])layer_sub.weight, (float (*)[6][6])layer_sub.back_pact);
	bkwd_pact_conv<<<64, 64>>>((float (*)[24][24])layer_conv.back_pact, (float (*)[24][24])layer_conv.back_out, (float (*)[24][24])layer_conv.pact);
	bkwd_weight_conv<<<64, 64>>>((float (*)[5][5])layer_conv.back_weight, (float (*)[24][24])layer_conv.back_pact, (float (*)[28])layer_input.output);
	bkwd_bias_conv<<<64, 64>>>(layer_conv.bias, (float (*)[24][24])layer_conv.back_pact);

	// apply the gradients to each layers weights/kernels
	grad<<<64, 64>>>(layer_full.weight, layer_full.back_weight, layer_full.size_kernel * layer_full.channels);
	grad<<<64, 64>>>(layer_sub.weight, layer_sub.back_weight, layer_sub.size_kernel * layer_sub.channels);
	grad<<<64, 64>>>(layer_conv.weight, layer_conv.back_weight, layer_conv.size_kernel * layer_conv.channels);

	// for timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return ms;
}

static void train()
{
	// for CUBLAS, creating a handle
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float total_error;
	int iterations = 50;
	float total_time = 0.0;
	while (iterations-- > 0) {
		total_error = 0.0;
		for (int i = 0; i < train_count; ++i) {
			float cur_err = 0;
			total_time += forward_pass(train_set[i].data);

			// reset to zero before doing backward pass
			layer_full.backprop_clear();
			layer_sub.backprop_clear();
			layer_conv.backprop_clear();

			// Euclid distance of train_set[i]
			calc_error<<<10, 1>>>(layer_full.back_pact, layer_full.output, train_set[i].label, 10);
			// 2 norm
			cublasSnrm2(blas, 10, layer_full.back_pact, 1, &cur_err);

			total_error += cur_err;

			total_time += back_pass();
		}
		total_error /= train_count;
		// output error and time on GPU
		fprintf(stdout, "Error: %e, GPU time: %lf\n", total_error, total_time);
		// stop training once threshold is passed
		if (total_error < threshold) {
			fprintf(stdout, "Finished Training!\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", total_time);
}


// Prediction function that forward passes data getting its label
static unsigned int predict(double data[28][28])
{
	// to get the output layer
	float output[10];

	// do the forward pass
	forward_pass(data);

	// keep track of max prob in output array
	// start at 0 for label 0
	unsigned int cur_max = 0;

	// copy the forward pass output to host
	cudaMemcpy(output, layer_full.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	// start at label 1
	for (int i = 1; i < 10; ++i) {
		if (output[cur_max] < output[i]) {
			cur_max = i;
		}
	}

	return cur_max;
}

// This function runs predictions on the test set to report test accuracy
static void run_test_set()
{
	// count hits
	int hit = 0;

	// iterate through the test set
	for (int i = 0; i < test_count; ++i) {
		if (predict(test_set[i].data) != test_set[i].label) {
			++hit;
		}
	}

	fprintf(stdout, "Accuracy: %.2lf%%\n", float(hit) / float(test_count) * 100.0);
}


int main()
{
	srand(time(NULL));
	std::cout<<"In main()!\n";
	CUresult total_error = cuInit(0);
	if (total_error != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", total_error);
		return 1;
	}

	std::cout<<"Loading data...\n";
	loaddata();
	std::cout<<"Beginning Training\n";
	train();
	std::cout<<"Running Test Set\n";
	run_test_set();

	return 0;
}