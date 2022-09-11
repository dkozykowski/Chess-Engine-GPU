#ifndef MACROS_H_INCLUDED
#define MARCOS_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>

typedef unsigned long long int pos64;

// Error handling

#define DEBUG 1
#define DBG(cmd) (if(DEBUG) cmd)

#define ERR(source) (perror(source),\
            fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
            exit(EXIT_FAILURE))

#define THREADS 1024
#define BLOCKS 1024

#define MAX_BOARDS_SIMULTANEOUSLY THREADS * BLOCKS
#define MAX_BOARDS_IN_MEMORY 1e6
#define MAX_DEPTH 16
#define BOARDS_GENERATED 64

#define INF 1e6;

#define BOARD_SIZE 64

#define WHITE 0
#define BLACK 1

// states
#define RIGHT 1
#define LEFT 2

#define WHITE_PAWN_STARTING_POS (pos64) 65280
#define WHITE_KNIGHT_STARTING_POS (pos64) 66
#define WHITE_BISHOP_STARTING_POS (pos64) 36
#define WHITE_ROOK_STARTING_POS (pos64) 129
#define WHITE_QUEEN_STARTING_POS (pos64) 16
#define WHITE_KING_STARTING_POS (pos64) 8

#define BLACk_PAWN_STARTING_POS (pos64) 71776119061217280
#define BLACK_KNIGHT_STARTING_POS (pos64) 4755801206503243776
#define BLACK_BISHOP_STARTING_POS (pos64) 2594073385365405696
#define BLACK_ROOK_STARTING_POS (pos64) 9295429630892703744
#define BLACK_QUEEN_STARTING_POS (pos64) 1152921504606846976
#define BLACK_KING_STARTING_POS (pos64) 576460752303423488

#define ALL_SET (pos64) 18446744073709551615


#endif