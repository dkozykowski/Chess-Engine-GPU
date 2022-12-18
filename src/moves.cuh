#ifndef MOVES_H_INCLUDED
#define MOVES_H_INCLUDED

#include "macros.cuh"

__host__ __device__ void generate_moves(pos64 *starting_boards, pos64 * generated_boards_space, bool isWhite);

__device__ int pre_count_moves(pos64 *starting_boards, bool isWhite);

#endif