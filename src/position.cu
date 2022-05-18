#include "position.cuh"

void print_position(char* board) {
    printf("+---+---+---+---+---+---+---+---+\n");
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            printf("| %c ", board[column + row * 8] == '.' ? ' ' : board[column + row * 8]);  
        }
        printf("|\n+---+---+---+---+---+---+---+---+\n");
    }
}

void flip_position(char* board) {
    for (int i = 0; i < 64; i++) {
        if ('a' <= board[i] && board[i] <= 'z') board[i] += 'A' - 'a';
        else if ('A' <= board[i] && board[i] <= 'Z') board[i] += 'a' - 'A';
    }
}

