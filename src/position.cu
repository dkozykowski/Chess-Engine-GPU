#include "position.cuh"

void printPosition(const pos64& whitePawns, const pos64& whiteBishops,
                   const pos64& whiteKnights, const pos64& whiteRooks,
                   const pos64& whiteQueens, const pos64& whiteKings,
                   const pos64& blackPawns, const pos64& blackBishops,
                   const pos64& blackKnights, const pos64& blackRooks,
                   const pos64& blackQueens, const pos64& blackKings) {
    printf(" +---+---+---+---+---+---+---+---+\n");

    for (int row = 7; row >= 0; row--) {
        for (int column = 0; column < 8; column++) {
            pos64 pos = ((pos64)1) << (column + (row << 3));
            char printableSign = ' ';

            if ((whitePawns & pos) != 0)
                printableSign = 'p';
            else if ((whiteBishops & pos) != 0)
                printableSign = 'b';
            else if ((whiteKnights & pos) != 0)
                printableSign = 'n';
            else if ((whiteRooks & pos) != 0)
                printableSign = 'r';
            else if ((whiteQueens & pos) != 0)
                printableSign = 'q';
            else if ((whiteKings & pos) != 0)
                printableSign = 'k';

            else if ((blackPawns & pos) != 0)
                printableSign = 'P';
            else if ((blackBishops & pos) != 0)
                printableSign = 'B';
            else if ((blackKnights & pos) != 0)
                printableSign = 'N';
            else if ((blackRooks & pos) != 0)
                printableSign = 'R';
            else if ((blackQueens & pos) != 0)
                printableSign = 'Q';
            else if ((blackKings & pos) != 0)
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

void swap(pos64& a, pos64& b) {
    pos64 c = a;
    a = b;
    b = c;
}

void flipPosition(pos64& whitePawns, pos64& whiteBishops, pos64& whiteKnights,
                  pos64& whiteRooks, pos64& whiteQueens, pos64& whiteKings,
                  pos64& blackPawns, pos64& blackBishops, pos64& blackKnights,
                  pos64& blackRooks, pos64& blackQueens, pos64& blackKings) {
    swap(whitePawns, blackPawns);
    swap(whiteBishops, blackBishops);
    swap(whiteKnights, blackKnights);
    swap(whiteRooks, blackRooks);
    swap(whiteQueens, blackQueens);
    swap(whiteKings, blackKings);
}

void moveChess(const int& fromCol, const int& fromRow, const int& toCol,
               const int& toRow, short& currentPlayer, pos64& whitePawns,
               pos64& whiteBishops, pos64& whiteKnights, pos64& whiteRooks,
               pos64& whiteQueens, pos64& whiteKings, pos64& blackPawns,
               pos64& blackBishops, pos64& blackKnights, pos64& blackRooks,
               pos64& blackQueens, pos64& blackKings) {
    pos64 from = ((pos64(1)) << (fromCol + (fromRow << 3)));
    pos64 to = ((pos64(1)) << (toCol + (toRow << 3)));

    pos64 toMask = (ALL_SET ^ to);

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
