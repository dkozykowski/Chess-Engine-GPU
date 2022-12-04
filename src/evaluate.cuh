#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include "macros.cuh"

// implementation of PeSTO's evaluation function 
// source: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

__device__ int  evaluate_position(const pos64& white_pawns,
                       const pos64& white_bishops,
                       const pos64& white_knights,
                       const pos64& white_rooks,
                       const pos64& white_queens,
                       const pos64& white_kings,
                       const pos64& black_pawns,
                       const pos64& black_bishops,
                       const pos64& black_knights,
                       const pos64& black_rooks,
                       const pos64& black_queens,
                       const pos64& black_kings);

#endif