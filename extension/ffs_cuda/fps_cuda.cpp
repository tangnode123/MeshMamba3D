// fps_cuda.cpp
#include <torch/extension.h>
#include <vector>

// Macro to check CUDA availability
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " Must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " Must be a contiguous tensor")

// Declare CUDA function
torch::Tensor farthest_point_sampling_cuda_forward(
    torch::Tensor dist_matrix,
    const int num_samples);

// C++ interface functions
torch::Tensor farthest_point_sampling_forward(
    torch::Tensor dist_matrix,
    const int num_samples) {
    
    // Check if the input is a CUDA tensor
    CHECK_CUDA(dist_matrix);
    
    // Check tensor data type
    CHECK_CONTIGUOUS(dist_matrix);
    TORCH_CHECK(dist_matrix.dtype() == torch::kInt32, "The distance matrix must be of type int32.");
    
    return farthest_point_sampling_cuda_forward(dist_matrix, num_samples);
}

// Register module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &farthest_point_sampling_forward, "FPS forward (CUDA)");
}