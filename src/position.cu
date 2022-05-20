#include "position.cuh"

void print_position(char* board) {
    printf(" +---+---+---+---+---+---+---+---+\n");
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            char printable_sign = PRINTABLE_POS[board[column + row * 8]];

            if (column == 0) 
                printf("%d| %c ", 8 - row, printable_sign); 
            else 
                printf("| %c ", printable_sign);   
            
        }
        printf("|\n +---+---+---+---+---+---+---+---+\n");
    }
    printf("   a   b   c   d   e   f   g   h\n");
}

void flip_position(char* board) {
    for (int i = 0; i < 64; i++) {
        if (board[i] != EMPTY)
            board[i] = OTHER(board[i]);
    }
}

