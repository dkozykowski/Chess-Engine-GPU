#ifndef MOVES_H_INCLUDED
#define MOVES_H_INCLUDED

#include "macros.cuh"

namespace MOVES {

__host__ __device__ void generateMoves(pos64 *startingBoards,
                                       pos64 *generatedBoardsSpace,
                                       bool isWhite);

__device__ int precountMoves(pos64 *startingBoards, bool isWhite);

} // namespace MOVES 

#endif // #ifndef MOVES_H_INCLUDED