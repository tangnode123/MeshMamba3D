// fps_cuda_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel function - Each thread processes one batch, using global memory
__global__ void farthest_point_sampling_kernel(
    const int* __restrict__ dist_matrix,
    int* __restrict__ samples,
    int* __restrict__ min_dist_global,  // Minimum distance array in global memory
    const int batch_size,
    const int num_points,
    const int num_samples) {
    
    // Calculate the current batch index
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size) return;
    
    // Calculate the starting positions of the distance matrix and sampling results for the current batch
    const int* dist_matrix_b = dist_matrix + b * num_points * num_points;
    int* samples_b = samples + b * num_samples;
    // Use global memory instead of shared memory 
    int* min_dist = min_dist_global + b * num_points;
    
    // Select the first sampling point (default is index 0)
    samples_b[0] = 0;
    
    // Initialize the minimum distance as the distance from the first point to all other points
    for (int i = 0; i < num_points; i++) {
        min_dist[i] = dist_matrix_b[samples_b[0] * num_points + i];
    }
    
    // Mark the selected points
    min_dist[samples_b[0]] = -1;
    
    // Sequentially select the remaining sampling points.
    for (int i = 1; i < num_samples; i++) {
        // Find the point corresponding to the maximum minimum distance.
        int max_dist = -1;
        int max_idx = 0;
        
        for (int j = 0; j < num_points; j++) {
            if (min_dist[j] > max_dist) {
                max_dist = min_dist[j];
                max_idx = j;
            }
        }
        
        // If no valid point is found, it means there are not enough points, so fill with the existing points.
        if (max_dist == -1) {
            samples_b[i] = samples_b[i-1];
            continue;
        }
        
        // Record the current sampling point.
        samples_b[i] = max_idx;
        
        // Mark this point as selected.
        min_dist[max_idx] = -1;
        
        // Update the minimum distance
        for (int j = 0; j < num_points; j++) {
            if (min_dist[j] != -1) {
                int new_dist = dist_matrix_b[max_idx * num_points + j];
                if (new_dist < min_dist[j]) {
                    min_dist[j] = new_dist;
                }
            }
        }
    }
}

// C++封装函数
torch::Tensor farthest_point_sampling_cuda_forward(
    torch::Tensor dist_matrix,
    const int num_samples) {
    
    // 获取维度信息
    const auto batch_size = dist_matrix.size(0);
    const auto num_points = dist_matrix.size(1);
    
    TORCH_CHECK(num_points == dist_matrix.size(2), "距离矩阵必须是方阵");
    TORCH_CHECK(num_samples > 0 && num_samples <= num_points, 
               "采样数量必须大于0且不超过点的总数");
    
    // 创建输出张量
    auto samples = torch::zeros({batch_size, num_samples}, 
                              torch::dtype(torch::kInt32).device(dist_matrix.device()));
    
    // 创建全局内存中的最小距离数组
    auto min_dist_global = torch::full({batch_size, num_points}, 
                                    std::numeric_limits<int>::max(),
                                    torch::dtype(torch::kInt32).device(dist_matrix.device()));
    
    // 计算线程块和线程数
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    // 调用CUDA核函数 - 不再使用共享内存
    farthest_point_sampling_kernel<<<blocks, threads>>>(
        dist_matrix.data_ptr<int>(),
        samples.data_ptr<int>(),
        min_dist_global.data_ptr<int>(),
        batch_size,
        num_points,
        num_samples
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA错误: ", cudaGetErrorString(err));
    
    return samples;
}