#include "moves.cuh"
#include "macros.cuh"

void make_move(char * board, 
                   int & current_player, 
                   int from_row, 
                   int from_col, 
                   int to_row, 
                   int to_col) {
    int from_pos = from_col + from_row * 8;
    int to_pos = to_col + to_row * 8;

    // check if current player owns the piece he tries to move
    if(board[from_pos] != EMPTY && PCOLOR(board[from_pos]) != current_player) {
        printf("Invalid move");
        return;
    }

    board[to_pos] = board[from_pos];
    board[from_pos] = EMPTY;
    current_player ^= 1;
}