#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <string>
#include <stdio.h>

#include "macros.cuh"

void print_position(const pos64& white_pawns,
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

void flip_position(pos64& white_pawns,
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

void move_chess(const int& from_col, 
                const int& from_row, 
                const int& to_col, 
                const int& to_row, 
                short& current_player,
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