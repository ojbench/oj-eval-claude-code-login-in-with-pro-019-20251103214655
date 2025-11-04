#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Cache K_all and V_all across iterations to avoid redundant concatenations
  Matrix* K_all_cache = nullptr;
  Matrix* V_all_cache = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move current_query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K_all incrementally from cache
    Matrix* K_all = nullptr;
    if (i == 0) {
      K_all = matrix_memory_allocator.Allocate("K_all");
      gpu_sim.Copy(keys[0], K_all, kInGpuHbm);
    } else {
      // Extend previous K_all with new key
      K_all = matrix_memory_allocator.Allocate("K_all");
      gpu_sim.Concat(K_all_cache, keys[i], K_all, 0, kInGpuHbm);
    }

    // Update cache for next iteration (but don't release yet)
    if (K_all_cache != nullptr) {
      gpu_sim.ReleaseMatrix(K_all_cache);
    }
    K_all_cache = matrix_memory_allocator.Allocate("K_cache");
    gpu_sim.Copy(K_all, K_all_cache, kInGpuHbm);

    // Move K_all to SRAM for faster operations
    gpu_sim.MoveMatrixToSharedMem(K_all);

    // Transpose K_all in SRAM
    gpu_sim.Transpose(K_all, kInSharedMemory);

    // Compute Q @ K^T
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_all, QK);
    gpu_sim.ReleaseMatrix(K_all);

    // Compute exp(QK)
    Matrix* exp_QK = matrix_memory_allocator.Allocate("exp_QK");
    gpu_sim.MatExp(QK, exp_QK);
    gpu_sim.ReleaseMatrix(QK);

    // Apply row-wise softmax
    Matrix* softmax_QK = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix* row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(exp_QK, r, row, kInSharedMemory);

      Matrix* sum = matrix_memory_allocator.Allocate("sum");
      gpu_sim.Sum(row, sum);

      Matrix* norm_row = matrix_memory_allocator.Allocate("norm");
      gpu_sim.MatDiv(row, sum, norm_row);
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(sum);

      if (r == 0) {
        softmax_QK = norm_row;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("temp_S");
        gpu_sim.Concat(softmax_QK, norm_row, temp, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        gpu_sim.ReleaseMatrix(norm_row);
        softmax_QK = temp;
      }
    }
    gpu_sim.ReleaseMatrix(exp_QK);

    // Build V_all incrementally from cache
    Matrix* V_all = nullptr;
    if (i == 0) {
      V_all = matrix_memory_allocator.Allocate("V_all");
      gpu_sim.Copy(values[0], V_all, kInGpuHbm);
    } else {
      V_all = matrix_memory_allocator.Allocate("V_all");
      gpu_sim.Concat(V_all_cache, values[i], V_all, 0, kInGpuHbm);
    }

    // Update cache for next iteration
    if (V_all_cache != nullptr) {
      gpu_sim.ReleaseMatrix(V_all_cache);
    }
    V_all_cache = matrix_memory_allocator.Allocate("V_cache");
    gpu_sim.Copy(V_all, V_all_cache, kInGpuHbm);

    // Move V_all to SRAM
    gpu_sim.MoveMatrixToSharedMem(V_all);

    // Compute attention output
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_QK, V_all, result);
    gpu_sim.ReleaseMatrix(softmax_QK);
    gpu_sim.ReleaseMatrix(V_all);

    // Move everything back to HBM
    gpu_sim.MoveMatrixToGpuHbm(current_query);
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