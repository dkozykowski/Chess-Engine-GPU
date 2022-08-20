#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "macros.cuh"

void init();

void terminate();

void search(const int& current_player,
            const int& move_num,
            const pos64& white_pawns,
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