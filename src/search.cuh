#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "macros.cuh"

void init();

void terminate();

void search(const int& current_player,
            const int& move_num,
            pos64& white_pawns,
            pos64& white_bishops,
            pos64& white_knights,
            pos64& white_rooks,
            pos64& white_queens,
            pos64& white_kings,
            pos64& black_pawns,
            pos64& black_bishops,
            pos64& black_knights,
            pos64& black_rooks,
            pos64& black_queens,
            pos64& black_kings);

#endif