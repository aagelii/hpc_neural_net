#include "layer.h"

// Constructor
Layer::Layer(int size_kernel, int channels, int input_size)
{
	// size_kernel = size of kernel
	// channels = number of channels
	// input_size = size of input
	this->size_kernel = size_kernel;
	this->channels = channels;
	this->input_size = input_size;

	// Allocate memory on CPU
	float host_bias[channels];
	float host_weight[channels][size_kernel];

	// For GPU
	output = NULL; // output
	pact = NULL; // pre-activation
	bias = NULL; // bias
	weight = NULL; // weight

	for (int i = 0; i < channels; ++i) {
		// host_bias[i] = 0.5 - float(rand()) / float(RAND_MAX); // Randomize bias
		host_bias[i] = 0.0; // if you didn't want to randomize bias, start from 0
		for (int j = 0; j < size_kernel; ++j) {
			// host_weight[i][j] = 0.5 - float(rand()) / float(RAND_MAX); // Randomize weight
			host_weight[i][j] = 0.0; // if you didn't want to randomize weight, start from 0
		}
	}

	// Allocate memory on GPU
	// for forward pass
	cudaMalloc(&output, sizeof(float) * input_size);
	cudaMalloc(&pact, sizeof(float) * input_size);
	cudaMalloc(&bias, sizeof(float) * channels);
	cudaMalloc(&weight, sizeof(float) * size_kernel * channels);

	// for backward pass
	cudaMalloc(&back_out, sizeof(float) * input_size);
	cudaMalloc(&back_pact, sizeof(float) * input_size);
	cudaMalloc(&back_weight, sizeof(float) * size_kernel * channels);

	// Copy data from CPU to GPU
	cudaMemcpy(bias, host_bias, sizeof(float) * channels, cudaMemcpyHostToDevice);
	cudaMemcpy(weight, host_weight, sizeof(float) * size_kernel * channels, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	// Free memory on GPU
	cudaFree(output);
	cudaFree(pact);
	cudaFree(bias);
	cudaFree(weight);
	cudaFree(back_out);
	cudaFree(back_pact);
	cudaFree(back_weight);
}

// Copying one data point from CPU to GPU
void Layer::load_image(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * input_size, cudaMemcpyHostToDevice);
}

// To clear out previous data from previous iterations, for forward pass
void Layer::clear()
{
	cudaMemset(output, 0, sizeof(float) * input_size);
	cudaMemset(pact, 0, sizeof(float) * input_size);
}

// To clear out previous data from previous iterations, for backward pass
void Layer::backprop_clear()
{
	cudaMemset(back_out, 0, sizeof(float) * input_size);
	cudaMemset(back_pact, 0, sizeof(float) * input_size);
	cudaMemset(back_weight, 0, sizeof(float) * size_kernel * channels);
}

// Sigmoid activation function
__device__ float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

// Applying activation function to input and storing it in output
__global__ void activation_function(float *input, float *output, const int N)
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Iterate over the input array, applying the sigmoid function to each element
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
		// Update the corresponding element in the output array
        output[idx] = sigmoid(input[idx]);
    }
}


// FORWARD PASS FUNCTION: For the first layer, calculating pre-activation, convolution layer
// Input image is 28x28, 6 kernels of grid_size 5x5
// Output is 6 feature maps of grid_size 24x24
__global__ void fwd_pact_conv(float input[28][28], float pact[6][24][24], float kernel[6][5][5])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the input array
    const int N = 5 * 5 * 6 * 24 * 24;

    // Iterate over the input array, applying the sigmoid function to each element
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		
        // Calculate the indices of the current element in the input array
        const int idx1 = (n % 5);
        const int idx2 = ((n /= 5) % 5);
        const int idx3 = ((n /= 5) % 6);
        const int idx4 = ((n /= 6) % 24);
        const int idx5 = ((n /= 24) % 24);

        // Update the corresponding element in the pact array
        atomicAdd(&pact[idx3][idx4][idx5], kernel[idx3][idx1][idx2] * input[idx4 + idx1][idx5 + idx2]);
    }
}


// FORWARD PASS FUNCTION: For the first layer, adding bias
__global__ void fwd_add_bias_conv(float pact[6][24][24], float bias[6])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the pact array
    const int N = 6 * 24 * 24;

    // Iterate over the pact array, adding the bias value to each element
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {

		// DONT CHANGE LOOP INVARIANT
		int n = idx;
        // Calculate the indices of the current element in the pact array
        const int idx1 = (n % 6);
        const int idx2 = ((n /= 6) % 24);
        const int idx3 = ((n /= 24) % 24);
        // Update the corresponding element in the pact array
        pact[idx1][idx2][idx3] += bias[idx1];
    }
}

// FORWARD PASS FUNCTION: For the second layer, calculating pre-activation, subsampling layer
// Input is 6 feature maps of grid_size 24x24, one window of grid_size 4x4
// Outputs 6 channels of grid_size 6x6
__global__ void fwd_pact_sub(float input[6][24][24], float pact[6][6][6], float window[1][4][4])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the input array
    const int N = 4 * 4 * 6 * 6 * 6;

    // Iterate over the input array, updating the corresponding element in the pact array
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
		
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
        // Calculate the indices of the current element in the input array
        const int idx1 = (n % 4);
        const int idx2 = ((n /= 4) % 4);
        const int idx3 = ((n /= 4) % 6);
        const int idx4 = ((n /= 6) % 6);
        const int idx5 = ((n /= 6) % 6);

        // Update the corresponding element in the pact array
        atomicAdd(&pact[idx3][idx4][idx5], window[0][idx1][idx2] * input[idx3][idx4 * 4 + idx1][idx5 * 4 + idx2]);
    }
}

// FORWARD PASS FUNCTION: For the second layer, adding bias
__global__ void fwd_add_bias_sub(float pact[6][6][6], float bias[1])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the input array
    const int N = 6 * 6 * 6;

    // Iterate over the input array, adding the bias value to each element
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
        // Calculate the indices of the current element in the input array
        const int idx1 = (n % 6);
        const int idx2 = ((n /= 6) % 6);
        const int idx3 = ((n /= 6) % 6);

        // Add the bias value to the current element
        pact[idx1][idx2][idx3] += bias[0];
    }
}


// FORWARD PASS FUNCTION: For the third layer, fully connected layer
// Input is 6 channels of grid_size 6x6, a weight matrix of grid_size 10x6x6x6
// Output is a vector of grid_size 10, the final layer
__global__ void fwd_pact_full(float input[6][6][6], float pact[10], float weight[10][6][6][6])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the input array
    const int N = 10 * 6 * 6 * 6;

    // Iterate over the input array, updating the corresponding element in the pact array
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
        // Calculate the indices of the current element in the input array
        const int idx1 = (n % 10);
        const int idx2 = ((n /= 10) % 6);
        const int idx3 = ((n /= 6) % 6);
        const int idx4 = ((n /= 6) % 6);

        // Update the corresponding element in the pact array
        atomicAdd(&pact[idx1], weight[idx1][idx2][idx3][idx4] * input[idx2][idx3][idx4]);
    }
}


// FORWARD PASS FUNCTION: For the third layer, adding bias
__global__ void fwd_add_bias_full(float pact[10], float bias[10])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the input array
    const int N = 10;

    // Iterate over the input array, adding the bias value to each element
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
        // Add the bias value to the current element
        pact[idx] += bias[idx];
    }
}

// BACKWARD PASS FUNCTION: For calculating error in the third layer
__global__ void calc_error(float *error, float *output, unsigned int label, const int N)
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Iterate over the elements in the error array
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
        // Calculate the error as the difference between the expected output
        // and the actual output of the neural network
        error[idx] = ((label == idx ? 1.0 : 0.0) - output[idx]);
    }
}

// BACKWARD PASS FUNCTION: For applying the gradient to the weights
__global__ void grad(float *weights, float *grads, const int N)
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Iterate over the elements in the weights array applying the gradient with the learning rate
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
        // Update the current element in the weights array
        // by adding the product of the learning rate and the corresponding
        // element in the grads array
        weights[idx] += learning_rate * grads[idx];
    }
}

// BACKWARD PASS FUNCTION: For calculating the gradients of the weights
__global__ void bkwd_weight_full(float back_weight[10][6][6][6], float back_pact[10], float output[6][6][6])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the back_weight array
    const int N = 10 * 6 * 6 * 6;

    // Iterate over the elements in the back_pact and output arrays, updating the corresponding element in the back_weight array
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
        // Calculate the indices of the current element in the back_weight array
        const int idx1 = (n % 10);
        const int idx2 = ((n /= 10) % 6);
        const int idx3 = ((n /= 6) % 6);
        const int idx4 = ((n /= 6) % 6);

        // Update the current element in the back_weight array
        back_weight[idx1][idx2][idx3][idx4] = back_pact[idx1] * output[idx2][idx3][idx4];
    }
}

// BACKWARD PASS FUNCTION: For applying the gradient to the bias of the third layer
__global__ void bkwd_bias_full(float bias[10], float back_pact[10])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in the bias array
    const int N = 10;

    // Iterate over the elements in the back_pact array applying the learning rate
    for (int idx = N * thread_pos / grid_size; 
         idx < N * (thread_pos+1) / grid_size; 
         ++idx) {
        // Update the current element in the bias array
        // by adding the product of the learning rate and the corresponding
        // element in the back_pact array
        bias[idx] += learning_rate * back_pact[idx];
    }
}

 // BACKWARD PASS FUNCTION: For calculating the gradients of the weights of the second layer
 __global__ void bkwd_output_sub(float back_out[6][6][6], float weight[10][6][6][6], float back_pact[10])
 {
	 // Calculate the current thread's position in the grid
	 const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	 // Calculate the total number of threads in the grid
	 const int grid_size = blockDim.x * gridDim.x;
 
	 // Calculate the total number of elements in the back_out array
	 const int N = 10 * 6 * 6 * 6;
 
	 // Iterate over the elements in the weights array and the back_pact array, updating the corresponding
	 for (int idx = N * thread_pos / grid_size; 
		  idx < N * (thread_pos+1) / grid_size; 
		  ++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		 // Calculate the indices of the current element in the back_out array
		 const int idx1 = (n % 10);
		 const int idx2 = ((n /= 10) % 6);
		 const int idx3 = ((n /= 6) % 6);
		 const int idx4 = ((n /= 6) % 6);
 
		 // Update the corresponding element in the back_out array
		 // by adding the product of the element in the weights array
		 // and the corresponding element in the back_pact array
		 atomicAdd(&back_out[idx2][idx3][idx4], weight[idx1][idx2][idx3][idx4] * back_pact[idx1]);
	 }
}
 
// BACKWARD PASS FUNCTION: For calculating the gradients of the weights of the second layer
__global__ void bkwd_pact_sub(float back_pact[6][6][6], float back_out[6][6][6], float pact[6][6][6])
{
    // Calculate the current thread's position in the grid
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    const int grid_size = blockDim.x * gridDim.x;

    // Calculate the total number of elements in back_pact
    const int N = 6 * 6 * 6;

    // Iterate over back_pact, applying the sigmoid derivative
    for (int idx = N * thread_pos / grid_size;
		 idx < N * (thread_pos+1) / grid_size;
		++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
        // Convert the index 'idx' into 3 indices for the back_pact array
        const int idx1 = (n % 6);
        const int idx2 = ((n /= 6) % 6);
        const int idx3 = ((n /= 6) % 6);

        // Calculate the output value of the sigmoid function
        const float o = sigmoid(pact[idx1][idx2][idx3]);

        // Calculate the derivative of the sigmoid function
        back_pact[idx1][idx2][idx3] = back_out[idx1][idx2][idx3] * o * (1 - o);
    }
}


// BACKWARD PASS FUNCTION: For calculating the gradients of the weights of the second layer
__global__ void bkwd_weight_sub(float back_weight[1][4][4], float back_pact[6][6][6], float output[6][24][24])
{
	// Calculate the current thread's position in the grid
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the total number of threads in the grid
	const int grid_size = blockDim.x * gridDim.x;

	// Calculate the total number of elements in the back_weight array
	const int N = 1 * 4 * 4 * 6 * 6 * 6;
	// Calculate the total number of elements in the back_pact array
	// const float d = pow(6.0, 3.0);

	// Iterate over the elements in the back_pact array and output array, updating the back_weight array
	for (int idx = N * thread_pos / grid_size; 
		idx < N * (thread_pos+1) / grid_size; 
		++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		// Calculate the indices of the current element in the back_out array
		const int idx1 = (n % 1);
		const int idx2 = (n % 4);
		const int idx3 = ((n /= 4) % 4);
		const int idx4 = ((n /= 4) % 6);
		const int idx5 = ((n /= 6) % 6);
		const int idx6 = ((n /= 6) % 6);

		// Update the corresponding element in the back_weight array
		// by adding the product of the element in the back_pact array
		// and the corresponding element in the output array
		atomicAdd(&back_weight[idx1][idx2][idx3], back_pact[idx4][idx5][idx6] * output[idx4][idx5 * 4 + idx2][idx6 * 4 + idx3]);
	}
}

// BACKWARD PASS FUNCTION: For applying the gradient to the bias of the second layer
__global__ void bkwd_bias_sub(float bias[1], float back_pact[6][6][6])
{
    // Calculate the current thread's position in the grid
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the total number of threads in the grid
	const int grid_size = blockDim.x * gridDim.x;

	// Calculate the total number of elements in the bias array
	const int N = 6 * 6 * 6;
	// Calculate the total number of elements in the back_pact array
	const float d = pow(6.0, 3.0);

	// Iterate over the elements in the back_pact array, applying the learning rate
	for (int idx = N * thread_pos / grid_size; 
		idx < N * (thread_pos+1) / grid_size; 
		++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		 // Calculate the indices of the current element in the back_pact array
		const int idx1 = (n % 6);
		const int idx2 = ((n /= 6) % 6);
		const int idx3 = ((n /= 6) % 6);

		// Update the corresponding element in the bias array
		// by adding the product of the element in the back_pact array
		// and the learning rate divided by the total number of elements in the back_pact array
		// Note: The learning rate is divided by the total number of elements in the back_pact array
		atomicAdd(&bias[0], learning_rate * back_pact[idx1][idx2][idx3] / d);
	}
}

 // BACKWARD PASS FUNCTION: For calculating the gradients of the weights of the first layer
__global__ void bkwd_output_conv(float back_out[6][24][24], float weight[1][4][4], float back_pact[6][6][6])
{
	// Calculate the current thread's position in the grid
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the total number of threads in the grid
	const int grid_size = blockDim.x * gridDim.x;

	// Calculate the total number of elements in the bias array
	const int N = 1*4*4*6*6*6;

	// Iterate over the elements in the weights array and back_pact array, updating the corresponding elements in the back_out array
	for (int idx = N * thread_pos / grid_size; 
		idx < N * (thread_pos+1) / grid_size; 
		++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		// Calculate the indices of the current element in the back_out array
		const int idx1 = (n % 1);
		const int idx2 = (n % 4);
		const int idx3 = ((n /= 4) % 4);
		const int idx4 = ((n /= 4) % 6);
		const int idx5 = ((n /= 6) % 6);
		const int idx6 = ((n /= 6) % 6);

		atomicAdd(&back_out[idx4][idx5 * 4 + idx2][idx6 * 4 + idx3], weight[idx1][idx2][idx3] * back_pact[idx4][idx5][idx6]);
	}
}

// BACKWARD PASS FUNCTION: For calculating the gradients of the weights of the first layer
__global__ void bkwd_pact_conv(float back_pact[6][24][24], float back_out[6][24][24], float pact[6][24][24])
{
	// Calculate the current thread's position in the grid
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the total number of threads in the grid
	const int grid_size = blockDim.x * gridDim.x;

	// Calculate the total number of elements in the bias array
	const int N = 6*24*24;

	// Iteracte over the elements in the back_out array, applying the sigmoid derivative function
	for (int idx = N * thread_pos / grid_size; idx < N * (thread_pos+1) / grid_size; ++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		// Calculate the indices of the current element in the back_pact array
		const int idx1 = (n % 6);
		const int idx2 = ((n /= 6) % 24);
		const int idx3 = ((n /= 24) % 24);

		// Calculate the sigmoid derivative of the corresponding element in the pact array
		const float o = sigmoid(pact[idx1][idx2][idx3]);

		back_pact[idx1][idx2][idx3] = back_out[idx1][idx2][idx3] * o * (1 - o);
	}
}

// BACKWARD PASS FUNCTION: For applying the gradient to the bias of the third layer
__global__ void bkwd_weight_conv(float back_weight[6][5][5], float back_pact[6][24][24], float output[28][28])
{
	// Calculate the current thread's position in the grid
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the total number of threads in the grid
	const int grid_size = blockDim.x * gridDim.x;

	// Calculate the total number of elements in the bias array
	const int N = 6 * 5 * 5 * 24 * 24;
	// Calculate the total number of elements in the back_pact array
	const float d = pow(24.0, 2.0);

	// Iterate over the elements in the back_pact array and output array, updating the corresponding elements in the back_weight array
	for (int idx = N * thread_pos / grid_size; 
		idx < N * (thread_pos+1) / grid_size; 
		++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		// Calculate the indices of the current element in the back_weight array
		const int idx1 = (n % 6);
		const int idx2 = ((n /= 6) % 5);
		const int idx3 = ((n /= 5) % 5);
		const int idx4 = ((n /= 5) % 24);
		const int idx5 = ((n /= 24) % 24);

		// Update the corresponding element in the back_weight array
		// Note: The division by d is to normalize the gradient
		atomicAdd(&back_weight[idx1][idx2][idx3], back_pact[idx1][idx4][idx5] * output[idx4 + idx2][idx5 + idx3] / d);
	}
}

// BACKWARD PASS FUNCTION: For applying the gradient to the bias of the first layer
__global__ void bkwd_bias_conv(float bias[6], float back_pact[6][24][24])
{
	// Calculate the current thread's position in the grid
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the total number of threads in the grid
	const int grid_size = blockDim.x * gridDim.x;

	// Calculate the total number of elements in the bias array
	const int N = 6 * 24 * 24;
	// Calculate the total number of elements in the back_pact array
	const float d = pow(24.0, 2.0);

	// Iterate over the elements in the back_pact array applying the learning rate
	for (int idx = N * thread_pos / grid_size; 
		idx < N * (thread_pos+1) / grid_size; 
		++idx) {
		// DONT CHANGE LOOP INVARIANT
		int n = idx;
		// Calculate the indices of the current element in the bias array
		const int idx1 = (n % 6);
		const int idx2 = ((n /= 6) % 24);
		const int idx3 = ((n /= 24) % 24);
		// Apply the learning rate to the corresponding element in the bias array
		// Note: The division by d is to normalize the gradient
		atomicAdd(&bias[idx1], learning_rate * back_pact[idx1][idx2][idx3] / d);
	}
}
