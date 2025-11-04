#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // In round i (loop index, 0-based), we use keys[0..i] and values[0..i]
    // Q has shape [i+1, d], each K and V has shape [1, d]
    // Attention formula: Softmax(Q @ K_all^T) @ V_all

    // Move current_query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Concatenate all keys[0..i] into K_all in HBM first
    Matrix* K_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        K_all = matrix_memory_allocator.Allocate("K_all");
        gpu_sim.Copy(keys[j], K_all, kInGpuHbm);
      } else {
        Matrix* temp_concat = matrix_memory_allocator.Allocate("temp_K");
        gpu_sim.Concat(K_all, keys[j], temp_concat, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(K_all);
        K_all = temp_concat;
      }
    }

    // Move K_all to SRAM and transpose: [i+1, d] -> [d, i+1]
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.Transpose(K_all, kInSharedMemory);

    // Compute Q @ K_all^T: [i+1, d] @ [d, i+1] = [i+1, i+1]
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_all, QK);
    gpu_sim.ReleaseMatrix(K_all);

    // Apply Softmax row-wise on QK
    // Compute exp(QK)
    Matrix* exp_QK = matrix_memory_allocator.Allocate("exp_QK");
    gpu_sim.MatExp(QK, exp_QK);
    gpu_sim.ReleaseMatrix(QK);

    // Build softmax matrix by processing each row
    Matrix* softmax_QK = nullptr;
    for (size_t row_idx = 0; row_idx <= i; ++row_idx) {
      Matrix* row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(exp_QK, row_idx, row, kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row, row_sum);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(row, row_sum, softmax_row);
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (row_idx == 0) {
        softmax_QK = softmax_row;
      } else {
        Matrix* temp_concat = matrix_memory_allocator.Allocate("temp_softmax");
        gpu_sim.Concat(softmax_QK, softmax_row, temp_concat, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax_QK = temp_concat;
      }
    }
    gpu_sim.ReleaseMatrix(exp_QK);

    // Concatenate all values[0..i] into V_all in HBM
    Matrix* V_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        V_all = matrix_memory_allocator.Allocate("V_all");
        gpu_sim.Copy(values[j], V_all, kInGpuHbm);
      } else {
        Matrix* temp_concat = matrix_memory_allocator.Allocate("temp_V");
        gpu_sim.Concat(V_all, values[j], temp_concat, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(V_all);
        V_all = temp_concat;
      }
    }

    // Move V_all to SRAM
    gpu_sim.MoveMatrixToSharedMem(V_all);

    // Compute final result: softmax_QK @ V_all = [i+1, i+1] @ [i+1, d] = [i+1, d]
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_QK, V_all, result);
    gpu_sim.ReleaseMatrix(softmax_QK);
    gpu_sim.ReleaseMatrix(V_all);

    // Move query back to HBM
    gpu_sim.MoveMatrixToGpuHbm(current_query);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu