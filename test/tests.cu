#include <gtest/gtest.h>
#include "../src/evaluate.cuh"
#include "../src/moves.cuh"
#include "../src/macros.cuh"
#include "../src/search.cuh"

__global__ void eval(int * result,
                pos64 white_pawns, pos64 white_bishops, pos64 white_knights, pos64 white_rooks, pos64 white_queens, pos64 white_kings,
                pos64 black_pawns, pos64 black_bishops, pos64 black_knights, pos64 black_rooks, pos64 black_queens, pos64 black_kings) {
    *result = evaluate_position(white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                          black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
}

void initBasePosition(pos64* board) 
{
    board[WHITE_PAWN_OFFSET] = WHITE_PAWN_STARTING_POS;
    board[WHITE_BISHOP_OFFSET] = WHITE_BISHOP_STARTING_POS;
    board[WHITE_KNIGHT_OFFSET] = WHITE_KNIGHT_STARTING_POS;
    board[WHITE_ROOK_OFFSET] = WHITE_ROOK_STARTING_POS;
    board[WHITE_QUEEN_OFFSET] = WHITE_QUEEN_STARTING_POS;
    board[WHITE_KING_OFFSET] = WHITE_KING_STARTING_POS;

    board[BLACK_PAWN_OFFSET] = BLACk_PAWN_STARTING_POS;
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
    int * d_result, * h_result;
    h_result = new int;
    cudaMalloc(&d_result, sizeof(int));
    eval<<<1, 1>>>(d_result, position[WHITE_PAWN_OFFSET], position[WHITE_BISHOP_OFFSET], position[WHITE_KNIGHT_OFFSET], position[WHITE_ROOK_OFFSET], position[WHITE_QUEEN_OFFSET], position[WHITE_KING_OFFSET], 
                          position[BLACK_PAWN_OFFSET], position[BLACK_BISHOP_OFFSET], position[BLACK_KNIGHT_OFFSET], position[BLACK_ROOK_OFFSET], position[BLACK_QUEEN_OFFSET], position[BLACK_KING_OFFSET]);
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    result = *h_result;
    delete h_result;
    cudaFree(d_result);

    // then
    ASSERT_EQ(0, result);
}

TEST(EvaluationTest, EndgamePositionEvaluation) {
    // given
    pos64 white_pawns = ((pos64) 1) << 5;
    pos64 white_bishops = 0;
    pos64 white_knights = 0;
    pos64 white_rooks = 0;
    pos64 white_queens = 0;
    pos64 white_kings = ((pos64) 1) << 13;

    pos64 black_pawns = ((pos64) 1) << 11 + ((pos64) 1) << 10;
    pos64 black_bishops = 0;
    pos64 black_knights = ((pos64) 1) << 40;
    pos64 black_rooks = 0;
    pos64 black_queens = ((pos64) 1) << 42;
    pos64 black_kings = ((pos64) 1) << 55;
    
    // when
    int result;
    int * d_result, * h_result;
    h_result = new int;
    cudaMalloc(&d_result, sizeof(int));
    eval<<<1, 1>>>(d_result, white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                          black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    result = *h_result;
    delete h_result;
    cudaFree(d_result);

    // then
    ASSERT_EQ(-1292, result);
}

TEST(GenerateMovesTest, StartPositionMovesSearch) {
    // given
    pos64 position[BOARD_SIZE];
    initBasePosition(position);
    
    short current_player = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if (((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] | (generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET]) == 0) break;
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(20, generated_moves_count);
}

TEST(SearchForBestMovesTest, StartPositionBestMoveSearchForWhite) {
    // given
    pos64 position[BOARD_SIZE];
    initBasePosition(position);

    short current_player = 0;
    // when
    init();
    search(current_player, position);
    terminate();

    pos64 basePosition[BOARD_SIZE];
    initBasePosition(basePosition);

    pos64 current_pos_white = position[WHITE_PAWN_OFFSET] | position[WHITE_BISHOP_OFFSET] | position[WHITE_KNIGHT_OFFSET] | position[WHITE_ROOK_OFFSET] | position[WHITE_QUEEN_OFFSET] | position[WHITE_KING_OFFSET];
    pos64 new_pos_white = basePosition[WHITE_PAWN_OFFSET] | basePosition[WHITE_BISHOP_OFFSET] | basePosition[WHITE_KNIGHT_OFFSET] | basePosition[WHITE_ROOK_OFFSET] | basePosition[WHITE_QUEEN_OFFSET] | basePosition[WHITE_KING_OFFSET];
    pos64 current_pos_black = position[BLACK_PAWN_OFFSET] | position[BLACK_BISHOP_OFFSET] | position[BLACK_KNIGHT_OFFSET] | position[BLACK_ROOK_OFFSET] | position[BLACK_QUEEN_OFFSET] | position[BLACK_KING_OFFSET];
    pos64 new_pos_black = basePosition[BLACK_PAWN_OFFSET] | basePosition[BLACK_BISHOP_OFFSET] | basePosition[BLACK_KNIGHT_OFFSET] | basePosition[BLACK_ROOK_OFFSET] | basePosition[BLACK_QUEEN_OFFSET] | basePosition[BLACK_KING_OFFSET];
    
    // then
    ASSERT_EQ(new_pos_black, current_pos_black);
    ASSERT_NE(new_pos_white, current_pos_white);
}

TEST(SearchForBestMovesTest, StartPositionBestMoveSearchForBlack) {
   // given
    pos64 position[BOARD_SIZE];
    initBasePosition(position);

    short current_player = 1;
    // when
    init();
    search(current_player, position);
    terminate();

    pos64 basePosition[BOARD_SIZE];
    initBasePosition(basePosition);

    pos64 current_pos_white = position[WHITE_PAWN_OFFSET] | position[WHITE_BISHOP_OFFSET] | position[WHITE_KNIGHT_OFFSET] | position[WHITE_ROOK_OFFSET] | position[WHITE_QUEEN_OFFSET] | position[WHITE_KING_OFFSET];
    pos64 new_pos_white = basePosition[WHITE_PAWN_OFFSET] | basePosition[WHITE_BISHOP_OFFSET] | basePosition[WHITE_KNIGHT_OFFSET] | basePosition[WHITE_ROOK_OFFSET] | basePosition[WHITE_QUEEN_OFFSET] | basePosition[WHITE_KING_OFFSET];
    pos64 current_pos_black = position[BLACK_PAWN_OFFSET] | position[BLACK_BISHOP_OFFSET] | position[BLACK_KNIGHT_OFFSET] | position[BLACK_ROOK_OFFSET] | position[BLACK_QUEEN_OFFSET] | position[BLACK_KING_OFFSET];
    pos64 new_pos_black = basePosition[BLACK_PAWN_OFFSET] | basePosition[BLACK_BISHOP_OFFSET] | basePosition[BLACK_KNIGHT_OFFSET] | basePosition[BLACK_ROOK_OFFSET] | basePosition[BLACK_QUEEN_OFFSET] | basePosition[BLACK_KING_OFFSET];
    
    // then
    ASSERT_NE(new_pos_black, current_pos_black);
    ASSERT_EQ(new_pos_white, current_pos_white);
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

    short current_player = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    int generated_attacks_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] < position[WHITE_PAWN_OFFSET]){ 
            generated_attacks_count++;
        }
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(8, generated_moves_count);
    ASSERT_EQ(2, generated_attacks_count);
}

TEST(GenerateMovesTest, WhiteKnightMovesTests) {

    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = 0;
    position[WHITE_BISHOP_OFFSET] = 0;
    position[WHITE_KNIGHT_OFFSET] = 68719476736;
    position[WHITE_ROOK_OFFSET] = 0;
    position[WHITE_QUEEN_OFFSET] = 0;
    position[WHITE_KING_OFFSET] = 0;

    position[BLACK_PAWN_OFFSET] = BLACk_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short current_player = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    int generated_attacks_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] < position[BLACK_PAWN_OFFSET]){ 
            generated_attacks_count++;
        }
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(8, generated_moves_count);
    ASSERT_EQ(2, generated_attacks_count);
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

    short current_player = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    int generated_attacks_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] < position[WHITE_PAWN_OFFSET]){ 
            generated_attacks_count++;
        }
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(13, generated_moves_count);
    ASSERT_EQ(1, generated_attacks_count);
}

TEST(GenerateMovesTest, WhiteRookMovesTests) {
    
    pos64 position[BOARD_SIZE];
    position[WHITE_PAWN_OFFSET] = 0;
    position[WHITE_BISHOP_OFFSET] = 0;
    position[WHITE_KNIGHT_OFFSET] = 0;
    position[WHITE_ROOK_OFFSET] = 68719476736;
    position[WHITE_QUEEN_OFFSET] = 0;
    position[WHITE_KING_OFFSET] = 0;

    position[BLACK_PAWN_OFFSET] = BLACk_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short current_player = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    int generated_attacks_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] < position[BLACK_PAWN_OFFSET]){ 
            generated_attacks_count++;
        }
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(13, generated_moves_count);
    ASSERT_EQ(1, generated_attacks_count);
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

    short current_player = 1;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    int generated_attacks_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] == 0) break;
        if((generatedMoves + x * BOARD_SIZE)[WHITE_KING_OFFSET] < position[WHITE_PAWN_OFFSET]){ 
            generated_attacks_count++;
        }
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(10, generated_moves_count);
    ASSERT_EQ(2, generated_attacks_count);
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

    position[BLACK_PAWN_OFFSET] = BLACk_PAWN_STARTING_POS;
    position[BLACK_BISHOP_OFFSET] = BLACK_BISHOP_STARTING_POS;
    position[BLACK_KNIGHT_OFFSET] = BLACK_KNIGHT_STARTING_POS;
    position[BLACK_ROOK_OFFSET] = BLACK_ROOK_STARTING_POS;
    position[BLACK_QUEEN_OFFSET] = BLACK_QUEEN_STARTING_POS;
    position[BLACK_KING_OFFSET] = BLACK_KING_STARTING_POS;

    short current_player = 0;

    pos64 generatedMoves[BOARDS_GENERATED * BOARD_SIZE];

    // when
    generate_moves(position, generatedMoves, current_player == WHITE);
    int generated_moves_count = 0;
    int generated_attacks_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((generatedMoves + x * BOARD_SIZE)[BLACK_KING_OFFSET] == 0) break;
        if((generatedMoves + x * BOARD_SIZE)[BLACK_PAWN_OFFSET] < position[BLACK_PAWN_OFFSET]){ 
            generated_attacks_count++;
        }
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(10, generated_moves_count);
    ASSERT_EQ(1, generated_attacks_count);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

