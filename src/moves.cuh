#ifndef MOVES_H_INCLUDED
#define MOVES_H_INCLUDED

#include <string>

void make_move(char * board, 
                   int & current_player, 
                   int from_row, 
                   int from_col, 
                   int to_row, 
                   int to_col);

#endif