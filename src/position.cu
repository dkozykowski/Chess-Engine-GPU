#include "position.cuh"

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
                    const pos64& black_kings) {
    printf(" +---+---+---+---+---+---+---+---+\n");

    for (int row = 7; row >= 0; row--) {
        for (int column = 0; column < 8; column++) {
            pos64 pos = ((pos64) 1) << (column + (row << 3));
            char printable_sign = ' ';

            if ((white_pawns & pos) != 0) printable_sign = 'p';
            else if ((white_bishops & pos) != 0) printable_sign = 'b';
            else if ((white_knights & pos) != 0) printable_sign = 'n';
            else if ((white_rooks & pos) != 0) printable_sign = 'r';
            else if ((white_queens & pos) != 0) printable_sign = 'q';
            else if ((white_kings & pos) != 0) printable_sign = 'k';

            else if ((black_pawns & pos) != 0) printable_sign = 'P';
            else if ((black_bishops & pos) != 0) printable_sign = 'B';
            else if ((black_knights & pos) != 0) printable_sign = 'N';
            else if ((black_rooks & pos) != 0) printable_sign = 'R';
            else if ((black_queens & pos) != 0) printable_sign = 'Q';
            else if ((black_kings & pos) != 0) printable_sign = 'K';

            if (column == 0) 
                printf("%d| %c ", 8 - row, printable_sign); 
            else 
                printf("| %c ", printable_sign); 

            pos >>= 1;
        }
        printf("|\n +---+---+---+---+---+---+---+---+\n");
    }
    printf("   a   b   c   d   e   f   g   h\n");
}

void swap(pos64& a, pos64& b) {
    pos64 c = a;
    a = b;
    b = c;
}

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
                   pos64& black_kings) {
    swap(white_pawns, black_pawns);
    swap(white_bishops, black_bishops);
    swap(white_knights, black_knights);
    swap(white_rooks, black_rooks);
    swap(white_queens, black_queens);
    swap(white_kings, black_kings);
}

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
                pos64& black_kings) {

    pos64 from = ((pos64(1)) << (from_col + (from_row << 3)));
    pos64 to =  ((pos64(1)) << (to_col + (to_row << 3)));

    pos64 to_mask = (ALL_SET ^  to);
    
    white_pawns &= to_mask;
    white_bishops &= to_mask;
    white_knights &= to_mask;
    white_rooks &= to_mask;
    white_queens &= to_mask;
    white_kings &= to_mask;

    black_pawns &= to_mask;
    black_bishops &= to_mask;
    black_knights &= to_mask;
    black_rooks &= to_mask;
    black_queens &= to_mask;
    black_kings &= to_mask;

    if (current_player == WHITE) {
        if ((white_pawns & from) != 0) {
            white_pawns ^= from;
            white_pawns |= to;
        }
        else if ((white_bishops & from) != 0) {
            white_bishops ^= from;
            white_bishops |= to;
        }
        else if ((white_knights & from) != 0) {
            white_knights ^= from;
            white_knights |= to;
        }
        else if ((white_rooks & from) != 0) {
            white_rooks ^= from;
            white_rooks |= to;
        }
        else if ((white_queens & from) != 0) {
            white_queens ^= from;
            white_queens |= to;
        }
        else if ((white_kings & from) != 0) {
            white_kings ^= from;
            white_kings |= to;
        }
    }
    else if (current_player == BLACK) {
        if ((black_pawns & from) != 0) {
            black_pawns ^= from;
            black_pawns |= to;
        }
        else if ((black_bishops & from) != 0) {
            black_bishops ^= from;
            black_bishops |= to;
        }
        else if ((black_knights & from) != 0) {
            black_knights ^= from;
            black_knights |= to;
        }
        else if ((black_rooks & from) != 0) {
            black_rooks ^= from;
            black_rooks |= to;
        }
        else if ((black_queens & from) != 0) {
            black_queens ^= from;
            black_queens |= to;
        }
        else if ((black_kings & from) != 0) {
            black_kings ^= from;
            black_kings |= to;
        }
    }
}
