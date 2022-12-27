#include <gtest/gtest.h>

#include "../src/evaluate.cuh"
#include "../src/macros.cuh"
#include "../src/moves.cuh"
#include "../src/search.cuh"

#define BOARDS_GENERATED 255

__global__ void eval(int* result, pos64 whitePawns, pos64 whiteBishops,
                     pos64 whiteKnights, pos64 whiteRooks, pos64 whiteQueens,
                     pos64 whiteKings, pos64 blackPawns, pos64 blackBishops,
                     pos64 blackKnights, pos64 blackRooks, pos64 blackQueens,
                     pos64 blackKings) {
    *result = EVALUATION::evaluatePosition(
        whitePawns, whiteBishops, whiteKnights, whiteRooks, whiteQueens,
        whiteKings, blackPawns, blackBishops, blackKnights, blackRooks,
        blackQueens, blackKings);
}

void initBasePosition(pos64* board) {
    board[WHITE_PAWN_OFFSET] = WHITE_PAWN_STARTING_POS;
    board[WHITE_BISHOP_OFFSET] = WHITE_BISHOP_STARTING_POS;
    board[WHITE_KNIGHT_OFFSET] = WHITE_KNIGHT_STARTING_POS;
    board[WHITE_ROOK_OFFSET] = WHITE_ROOK_STARTING_POS;
    board[WHITE_QUEEN_OFFSET] = WHITE_QUEEN_STARTING_POS;
    board[WHITE_KING_OFFSET] = WHITE_KING_STARTING_POS;

    board[BLACK_PAWN_OFFSET] = BLACK_PAWN_STARTING_POS;
    board[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    board[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    board[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    board[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    board[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;
}

TEST(EvaluationTest, StartPositionEvaluation) {
    // given

    pos64 position[BOARD_SIZE];
    initBasePosition(position);

    // when
    int result;
    int *dResult, *hResult;
    hResult = new int;
    cudaMalloc(&dResult, sizeof(int));
    eval<<<1, 1>>>(dResult, position[WHITE_PAWN_OFFSET],
                   position[WHITE_BISHOP_OFFSET], position[WHITE_KNIGHT_OFFSET],
                   position[WHITE_ROOK_OFFSET], position[WHITE_QUEEN_OFFSET],
                   position[WHITE_KING_OFFSET], position[BLACK_PAWN_OFFSET],
                   position[BLACK_BISHOP_OFFSET], position[BLACK_KNIGHT_OFFSET],
                   position[BLACK_ROOK_OFFSET], position[BLACK_QUEEN_OFFSET],
                   position[BLACK_KING_OFFSET]);
    cudaMemcpy(hResult, dResult, sizeof(int), cudaMemcpyDeviceToHost);
    result = *hResult;
    delete hResult;
    cudaFree(dResult);

    // then
    ASSERT_EQ(0, result);
}

TEST(EvaluationTest, EndgamePositionEvaluation) {
    // given
    pos64 whitePawns = ((pos64)1) << 5;
    pos64 whiteBishops = 0;
    pos64 whiteKnights = 0;
    pos64 whiteRooks = 0;
    pos64 whiteQueens = 0;
    pos64 whiteKings = ((pos64)1) << 13;

    pos64 blackPawns = ((pos64)1) << 11 + ((pos64)1) << 10;
    pos64 blackBishops = 0;
    pos64 blackKnights = ((pos64)1) << 40;
    pos64 blackRooks = 0;
    pos64 blackQueens = ((pos64)1) << 42;
    pos64 blackKings = ((pos64)1) << 55;

    // when
    int result;
    int *dResult, *hResult;
    hResult = new int;
    cudaMalloc(&dResult, sizeof(int));
    eval<<<1, 1>>>(dResult, whitePawns, whiteBishops, whiteKnights, whiteRooks,
                   whiteQueens, whiteKings, blackPawns, blackBishops,
                   blackKnights, blackRooks, blackQueens, blackKings);
    cudaMemcpy(hResult, dResult, sizeof(int), cudaMemcpyDeviceToHost);
    result = *hResult;
    delete hResult;
    cudaFree(dResult);

    // then
    ASSERT_EQ(-1292, result);
}

TEST(GenerateMovesTest, StartPositionMovesSearch) {
    // given
    pos64 position[BOARD_SIZE];
    initBasePosition(position);

    short currentPlayer = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if (((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] |
             (generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET]) == 0)
            break;
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(20, generatedMovesCount);
}

TEST(SearchForBestMovesTest, StartPositionBestMoveSearchForWhite) {
    // given
    pos64 position[BOARD_SIZE];
    initBasePosition(position);

    short currentPlayer = 0;
    // when
    SEARCH::init();
    SEARCH::findBestMove(currentPlayer, position);
    SEARCH::terminate();

    pos64 basePosition[BOARD_SIZE];
    initBasePosition(basePosition);

    pos64 currentPosWhite =
        position[WHITE_PAWN_OFFSET] | position[WHITE_BISHOP_OFFSET] |
        position[WHITE_KNIGHT_OFFSET] | position[WHITE_ROOK_OFFSET] |
        position[WHITE_QUEEN_OFFSET] | position[WHITE_KING_OFFSET];
    pos64 newPosWhite =
        basePosition[WHITE_PAWN_OFFSET] | basePosition[WHITE_BISHOP_OFFSET] |
        basePosition[WHITE_KNIGHT_OFFSET] | basePosition[WHITE_ROOK_OFFSET] |
        basePosition[WHITE_QUEEN_OFFSET] | basePosition[WHITE_KING_OFFSET];
    pos64 currentPosBlack =
        position[BLACK_PAWN_OFFSET] | position[BLACK_BISHOP_OFFSET] |
        position[BLACK_KNIGHT_OFFSET] | position[BLACK_ROOK_OFFSET] |
        position[BLACK_QUEEN_OFFSET] | position[BLACK_KING_OFFSET];
    pos64 newPosBlack =
        basePosition[BLACK_PAWN_OFFSET] | basePosition[BLACK_BISHOP_OFFSET] |
        basePosition[BLACK_KNIGHT_OFFSET] | basePosition[BLACK_ROOK_OFFSET] |
        basePosition[BLACK_QUEEN_OFFSET] | basePosition[BLACK_KING_OFFSET];

    // then
    ASSERT_EQ(newPosBlack, currentPosBlack);
    ASSERT_NE(newPosWhite, currentPosWhite);
}

TEST(SearchForBestMovesTest, StartPositionBestMoveSearchForBlack) {
    // given
    pos64 position[BOARD_SIZE];
    initBasePosition(position);

    short currentPlayer = 1;
    // when
    SEARCH::init();
    SEARCH::findBestMove(currentPlayer, position);
    SEARCH::terminate();

    pos64 basePosition[BOARD_SIZE];
    initBasePosition(basePosition);

    pos64 currentPosWhite =
        position[WHITE_PAWN_OFFSET] | position[WHITE_BISHOP_OFFSET] |
        position[WHITE_KNIGHT_OFFSET] | position[WHITE_ROOK_OFFSET] |
        position[WHITE_QUEEN_OFFSET] | position[WHITE_KING_OFFSET];
    pos64 newPosWhite =
        basePosition[WHITE_PAWN_OFFSET] | basePosition[WHITE_BISHOP_OFFSET] |
        basePosition[WHITE_KNIGHT_OFFSET] | basePosition[WHITE_ROOK_OFFSET] |
        basePosition[WHITE_QUEEN_OFFSET] | basePosition[WHITE_KING_OFFSET];
    pos64 currentPosBlack =
        position[BLACK_PAWN_OFFSET] | position[BLACK_BISHOP_OFFSET] |
        position[BLACK_KNIGHT_OFFSET] | position[BLACK_ROOK_OFFSET] |
        position[BLACK_QUEEN_OFFSET] | position[BLACK_KING_OFFSET];
    pos64 newPosBlack =
        basePosition[BLACK_PAWN_OFFSET] | basePosition[BLACK_BISHOP_OFFSET] |
        basePosition[BLACK_KNIGHT_OFFSET] | basePosition[BLACK_ROOK_OFFSET] |
        basePosition[BLACK_QUEEN_OFFSET] | basePosition[BLACK_KING_OFFSET];

    // then
    ASSERT_NE(newPosBlack, currentPosBlack);
    ASSERT_EQ(newPosWhite, currentPosWhite);
}

TEST(GenerateMovesTest, BlackKnightMovesTests) {
    // given
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = WHITE_PAWN_STARTING_POS;
    position[WHITE_BISHOP_OFFSET] = WHITE_BISHOP_STARTING_POS;
    position[WHITE_KNIGHT_OFFSET] = WHITE_KNIGHT_STARTING_POS;
    position[WHITE_ROOK_OFFSET] = WHITE_ROOK_STARTING_POS;
    position[WHITE_QUEEN_OFFSET] = WHITE_QUEEN_STARTING_POS;
    position[WHITE_KING_OFFSET] = WHITE_KING_STARTING_POS;

    position[BLACK_PAWN_OFFSET] = 0;
    position[BLACK_BISHOP_OFFSET] = 0;
    position[BLACK_KNIGHT_OFFSET] = 536870912;
    position[BLACK_ROOK_OFFSET] = 0;
    position[BLACK_QUEEN_OFFSET] = 0;
    position[BLACK_KING_OFFSET] = 0;

    short currentPlayer = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_PAWN_OFFSET] <
            position[WHITE_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(8, generatedMovesCount);
    ASSERT_EQ(2, generatedAttacksCount);
}

TEST(GenerateMovesTest, WhiteKnightMovesTests) {
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = 0;
    position[WHITE_BISHOP_OFFSET] = 0;
    position[WHITE_KNIGHT_OFFSET] = 68719476736;
    position[WHITE_ROOK_OFFSET] = 0;
    position[WHITE_QUEEN_OFFSET] = 0;
    position[WHITE_KING_OFFSET] = 0;

    position[BLACK_PAWN_OFFSET] = BLACK_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short currentPlayer = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] <
            position[BLACK_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(8, generatedMovesCount);
    ASSERT_EQ(2, generatedAttacksCount);
}

TEST(GenerateMovesTest, BlackRookMovesTests) {
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = WHITE_PAWN_STARTING_POS;
    position[WHITE_BISHOP_OFFSET] = WHITE_BISHOP_STARTING_POS;
    position[WHITE_KNIGHT_OFFSET] = WHITE_KNIGHT_STARTING_POS;
    position[WHITE_ROOK_OFFSET] = WHITE_ROOK_STARTING_POS;
    position[WHITE_QUEEN_OFFSET] = WHITE_QUEEN_STARTING_POS;
    position[WHITE_KING_OFFSET] = WHITE_KING_STARTING_POS;

    position[BLACK_PAWN_OFFSET] = 0;
    position[BLACK_BISHOP_OFFSET] = 0;
    position[BLACK_KNIGHT_OFFSET] = 0;
    position[BLACK_ROOK_OFFSET] = 536870912;
    position[BLACK_QUEEN_OFFSET] = 0;
    position[BLACK_KING_OFFSET] = 0;

    short currentPlayer = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_PAWN_OFFSET] <
            position[WHITE_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(13, generatedMovesCount);
    ASSERT_EQ(1, generatedAttacksCount);
}

TEST(GenerateMovesTest, WhiteRookMovesTests) {
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = 0;
    position[WHITE_BISHOP_OFFSET] = 0;
    position[WHITE_KNIGHT_OFFSET] = 0;
    position[WHITE_ROOK_OFFSET] = 68719476736;
    position[WHITE_QUEEN_OFFSET] = 0;
    position[WHITE_KING_OFFSET] = 0;

    position[BLACK_PAWN_OFFSET] = BLACK_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short currentPlayer = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] <
            position[BLACK_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(13, generatedMovesCount);
    ASSERT_EQ(1, generatedAttacksCount);
}

TEST(GenerateMovesTest, BlackBishopMovesTests) {
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = WHITE_PAWN_STARTING_POS;
    position[WHITE_BISHOP_OFFSET] = WHITE_BISHOP_STARTING_POS;
    position[WHITE_KNIGHT_OFFSET] = WHITE_KNIGHT_STARTING_POS;
    position[WHITE_ROOK_OFFSET] = WHITE_ROOK_STARTING_POS;
    position[WHITE_QUEEN_OFFSET] = WHITE_QUEEN_STARTING_POS;
    position[WHITE_KING_OFFSET] = WHITE_KING_STARTING_POS;

    position[BLACK_PAWN_OFFSET] = 0;
    position[BLACK_BISHOP_OFFSET] = 536870912;
    position[BLACK_KNIGHT_OFFSET] = 0;
    position[BLACK_ROOK_OFFSET] = 0;
    position[BLACK_QUEEN_OFFSET] = 0;
    position[BLACK_KING_OFFSET] = 0;

    short currentPlayer = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_PAWN_OFFSET] <
            position[WHITE_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(10, generatedMovesCount);
    ASSERT_EQ(2, generatedAttacksCount);
}

TEST(GenerateMovesTest, WhiteBishopMovesTests) {
    // given
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = 0;
    position[WHITE_BISHOP_OFFSET] = 68719476736;
    position[WHITE_KNIGHT_OFFSET] = 0;
    position[WHITE_ROOK_OFFSET] = 0;
    position[WHITE_QUEEN_OFFSET] = 0;
    position[WHITE_KING_OFFSET] = 0;

    position[BLACK_PAWN_OFFSET] = BLACK_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short currentPlayer = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] <
            position[BLACK_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(11, generatedMovesCount);
    ASSERT_EQ(2, generatedAttacksCount);
}

TEST(GenerateMovesTest, BlackQueenMovesTests) {
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = WHITE_PAWN_STARTING_POS;
    position[WHITE_BISHOP_OFFSET] = WHITE_BISHOP_STARTING_POS;
    position[WHITE_KNIGHT_OFFSET] = WHITE_KNIGHT_STARTING_POS;
    position[WHITE_ROOK_OFFSET] = WHITE_ROOK_STARTING_POS;
    position[WHITE_QUEEN_OFFSET] = WHITE_QUEEN_STARTING_POS;
    position[WHITE_KING_OFFSET] = WHITE_KING_STARTING_POS;

    position[BLACK_PAWN_OFFSET] = 0;
    position[BLACK_BISHOP_OFFSET] = 0;
    position[BLACK_KNIGHT_OFFSET] = 0;
    position[BLACK_ROOK_OFFSET] = 0;
    position[BLACK_QUEEN_OFFSET] = 536870912;
    position[BLACK_KING_OFFSET] = 0;

    short currentPlayer = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_PAWN_OFFSET] <
            position[WHITE_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(23, generatedMovesCount);
    ASSERT_EQ(3, generatedAttacksCount);
}

TEST(GenerateMovesTest, WhiteQueenMovesTests) {
    // given
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = 0;
    position[WHITE_BISHOP_OFFSET] = 0;
    position[WHITE_KNIGHT_OFFSET] = 0;
    position[WHITE_ROOK_OFFSET] = 0;
    position[WHITE_QUEEN_OFFSET] = 68719476736;
    position[WHITE_KING_OFFSET] = 0;

    position[BLACK_PAWN_OFFSET] = BLACK_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short currentPlayer = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    for (int i = 0; i < BOARDS_GENERATED * BOARD_SIZE; i++) {
        generatedMoves[i] = 0;
    }

    // when
    MOVES::generateMoves(position, generatedMoves, currentPlayer == WHITE);
    int generatedMovesCount = 0;
    int generatedAttacksCount = 0;
    for (int x = 0; x < BOARDS_GENERATED; x++) {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] <
            position[BLACK_PAWN_OFFSET]) {
            generatedAttacksCount++;
        }
        generatedMovesCount = x + 1;
    }

    // then
    ASSERT_EQ(24, generatedMovesCount);
    ASSERT_EQ(3, generatedAttacksCount);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
