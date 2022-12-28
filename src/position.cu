#include "position.cuh"

namespace POSITION {

/**
 * Pretty-prints the given board on the console.
 *
 * @param position Pointer to array storing the positions of chess pieces.
 */
void printPosition(pos64 *position) {
    printf(" +---+---+---+---+---+---+---+---+\n");

    for (int row = 7; row >= 0; row--) {
        for (int column = 0; column < 8; column++) {
            pos64 pos = ((pos64)1) << (column + (row << 3));
            char printableSign = ' ';

            if ((position[WHITE_PAWN_OFFSET] & pos) != 0)
                printableSign = 'p';
            else if ((position[WHITE_BISHOP_OFFSET] & pos) != 0)
                printableSign = 'b';
            else if ((position[WHITE_KNIGHT_OFFSET] & pos) != 0)
                printableSign = 'n';
            else if ((position[WHITE_ROOK_OFFSET] & pos) != 0)
                printableSign = 'r';
            else if ((position[WHITE_QUEEN_OFFSET] & pos) != 0)
                printableSign = 'q';
            else if ((position[WHITE_KING_OFFSET] & pos) != 0)
                printableSign = 'k';

            else if ((position[BLACK_PAWN_OFFSET] & pos) != 0)
                printableSign = 'P';
            else if ((position[BLACK_BISHOP_OFFSET] & pos) != 0)
                printableSign = 'B';
            else if ((position[BLACK_KNIGHT_OFFSET] & pos) != 0)
                printableSign = 'N';
            else if ((position[BLACK_ROOK_OFFSET] & pos) != 0)
                printableSign = 'R';
            else if ((position[BLACK_QUEEN_OFFSET] & pos) != 0)
                printableSign = 'Q';
            else if ((position[BLACK_KING_OFFSET] & pos) != 0)
                printableSign = 'K';

            if (column == 0)
                printf("%d| %c ", row + 1, printableSign);
            else
                printf("| %c ", printableSign);

            pos >>= 1;
        }
        printf("|\n +---+---+---+---+---+---+---+---+\n");
    }
    printf("   a   b   c   d   e   f   g   h\n");
}

void swap(pos64* a, pos64* b) {
    pos64 c = *a;
    *a = *b;
    *b = c;
}

/**
 * Flips the colors of pieces to the opposite.
 *
 * @param[out] position Pointer to array of pointers pointing to place in memory with positions of chess pieces.
 */
void flipPosition(pos64 **position) {
    swap(position[WHITE_PAWN_OFFSET], position[BLACK_PAWN_OFFSET]);
    swap(position[WHITE_BISHOP_OFFSET], position[BLACK_BISHOP_OFFSET]);
    swap(position[WHITE_KNIGHT_OFFSET], position[BLACK_KNIGHT_OFFSET]);
    swap(position[WHITE_ROOK_OFFSET], position[BLACK_ROOK_OFFSET]);
    swap(position[WHITE_QUEEN_OFFSET], position[BLACK_QUEEN_OFFSET]);
    swap(position[WHITE_KING_OFFSET], position[BLACK_KING_OFFSET]);
}

/**
 * Changes the position of a specific piece on the board.
 *
 * @param fromCol Index of column where the piece to move is.
 * @param fromRow Index of row where the piece to move is.
 * @param toCol Index of column where the piece will be moved to.
 * @param toRow Index of row where the piece will be moved to.
 * @param[out] position Pointer to array of pointers pointing to place in memory with positions of chess pieces.
 */
void moveChess(const int& fromCol, const int& fromRow, const int& toCol,
               const int& toRow, short& currentPlayer, pos64& whitePawns,
               pos64 **position) {
    pos64 from = ((pos64(1)) << (fromCol + (fromRow << 3)));
    pos64 to = ((pos64(1)) << (toCol + (toRow << 3)));

    pos64 toMask = (ALL_SET ^ to);

    pos64& whitePawns = *position[WHITE_PAWN_OFFSET];
    pos64& whiteBishops = *position[WHITE_BISHOP_OFFSET];
    pos64& whiteKnights = *position[WHITE_KNIGHT_OFFSET];
    pos64& whiteRooks = *position[WHITE_ROOK_OFFSET];
    pos64& whiteQueens = *position[WHITE_QUEEN_OFFSET];
    pos64& whiteKings = *position[WHITE_KING_OFFSET];

    pos64& blackPawns = *position[BLACK_PAWN_OFFSET];
    pos64& blackBishops = *position[BLACK_BISHOP_OFFSET];
    pos64& blackKnights = *position[BLACK_KNIGHT_OFFSET];
    pos64& blackRooks = *position[BLACK_ROOK_OFFSET];
    pos64& blackQueens = *position[BLACK_QUEEN_OFFSET];
    pos64& blackKings = *position[BLACK_KING_OFFSET];

    whitePawns &= toMask;
    whiteBishops &= toMask;
    whiteKnights &= toMask;
    whiteRooks &= toMask;
    whiteQueens &= toMask;
    whiteKings &= toMask;

    blackPawns &= toMask;
    blackBishops &= toMask;
    blackKnights &= toMask;
    blackRooks &= toMask;
    blackQueens &= toMask;
    blackKings &= toMask;

    if (currentPlayer == WHITE) {
        if ((whitePawns & from) != 0) {
            whitePawns ^= from;
            whitePawns |= to;
        } else if ((whiteBishops & from) != 0) {
            whiteBishops ^= from;
            whiteBishops |= to;
        } else if ((whiteKnights & from) != 0) {
            whiteKnights ^= from;
            whiteKnights |= to;
        } else if ((whiteRooks & from) != 0) {
            whiteRooks ^= from;
            whiteRooks |= to;
        } else if ((whiteQueens & from) != 0) {
            whiteQueens ^= from;
            whiteQueens |= to;
        } else if ((whiteKings & from) != 0) {
            whiteKings ^= from;
            whiteKings |= to;
        }
    } else if (currentPlayer == BLACK) {
        if ((blackPawns & from) != 0) {
            blackPawns ^= from;
            blackPawns |= to;
        } else if ((blackBishops & from) != 0) {
            blackBishops ^= from;
            blackBishops |= to;
        } else if ((blackKnights & from) != 0) {
            blackKnights ^= from;
            blackKnights |= to;
        } else if ((blackRooks & from) != 0) {
            blackRooks ^= from;
            blackRooks |= to;
        } else if ((blackQueens & from) != 0) {
            blackQueens ^= from;
            blackQueens |= to;
        } else if ((blackKings & from) != 0) {
            blackKings ^= from;
            blackKings |= to;
        }
    }
}
}  // namespace POSITION