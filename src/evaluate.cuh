#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include "macros.cuh"

// implementation of PeSTO's evaluation function
// source: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

namespace EVALUATION {

__device__ int evaluatePosition(
    const pos64& whitePawns, const pos64& whiteBishops,
    const pos64& whiteKnights, const pos64& whiteRooks,
    const pos64& whiteQueens, const pos64& whiteKings, const pos64& blackPawns,
    const pos64& blackBishops, const pos64& blackKnights,
    const pos64& blackRooks, const pos64& blackQueens, const pos64& blackKings);

}  // namespace EVALUATION

#endif  // #ifndef EVALUATE_H_INCLUDED