#ifndef MACROS_H_INCLUDED
#define MARCOS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>

typedef unsigned long long int pos64;

#define DEBUG 0
#define DEBUG2 1
#define DBG(cmd) \
    if (DEBUG) cmd
#define DBG2(cmd) \
    if (DEBUG2) cmd
#define CHECK_ALLOC(alloc)                                                    \
    {                                                                         \
        cudaError_t cu_err;                                                   \
        if ((cu_err = alloc) != cudaSuccess) ERR(cudaGetErrorString(cu_err)); \
    }
#define ERR(source)                                                  \
    (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), \
     exit(EXIT_FAILURE))

void gpuAssert(cudaError_t code, const char *file, int line);

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

#define MAX_THREADS 256
#define MAX_BLOCKS_PER_DIMENSION 65535

#define MAX_BOARD_COMPUTED_IN_SECOUND_STAGE 400
#define MAX_POSSIBLE_DEPTH 10

#define MAX_DEPTH 10

#define INF 1e6

#define WHITE 0
#define BLACK 1

// states
#define RIGHT 1
#define LEFT 2

#define STARTING_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

#define BLACK_LAST_LANE (pos64)255
#define WHITE_LAST_LANE (pos64)18374686479671624000

#define ALL_SET (pos64)18446744073709551615

#define NOT_A_FILE (pos64)0xfefefefefefefefe
#define NOT_H_FILE (pos64)0x7f7f7f7f7f7f7f7f

#define WHITE_PAWN_OFFSET 0
#define WHITE_KNIGHT_OFFSET 1
#define WHITE_BISHOP_OFFSET 2
#define WHITE_ROOK_OFFSET 3
#define WHITE_QUEEN_OFFSET 4
#define WHITE_KING_OFFSET 5

#define BLACK_PAWN_OFFSET 6
#define BLACK_KNIGHT_OFFSET 7
#define BLACK_BISHOP_OFFSET 8
#define BLACK_ROOK_OFFSET 9
#define BLACK_QUEEN_OFFSET 10
#define BLACK_KING_OFFSET 11

#define BOARD_SIZE 12

#endif