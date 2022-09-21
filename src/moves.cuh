#ifndef MOVES_H_INCLUDED
#define MOVES_H_INCLUDED

#include "macros.cuh"

__host__ __device__ void generate_moves(pos64 * white_pawns_boards,
                    pos64 * white_bishops_boards,
                    pos64 * white_knights_boards,
                    pos64 * white_rooks_boards,
                    pos64 * white_queens_boards,
                    pos64 * white_kings_boards,
                    pos64 * black_pawns_boards,
                    pos64 * black_bishops_boards,
                    pos64 * black_knights_boards,
                    pos64 * black_rooks_boards,
                    pos64 * black_queens_boards,
                    pos64 * black_kings_boards,
                    int * results,
                    int * depths,
                    short * stack_states,
                    int depth);

#endif