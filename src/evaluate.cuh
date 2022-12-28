#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include "macros.cuh"

// implementation of PeSTO's evaluation function
// source: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

namespace EVALUATION {

__device__ int evaluatePosition(pos64 *position);

}  // namespace EVALUATION

#endif  // #ifndef EVALUATE_H_INCLUDED