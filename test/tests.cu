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

TEST(EvaluationTest, StartPositionEvaluation) {
    // given
    pos64 white_pawns = WHITE_PAWN_STARTING_POS;
    pos64 white_bishops = WHITE_BISHOP_STARTING_POS;
    pos64 white_knights = WHITE_KNIGHT_STARTING_POS;
    pos64 white_rooks = WHITE_ROOK_STARTING_POS;
    pos64 white_queens = WHITE_QUEEN_STARTING_POS;
    pos64 white_kings = WHITE_KING_STARTING_POS;

    pos64 black_pawns = BLACk_PAWN_STARTING_POS;
    pos64 black_bishops = BLACK_BISHOP_STARTING_POS;
    pos64 black_knights = BLACK_KNIGHT_STARTING_POS;
    pos64 black_rooks = BLACK_ROOK_STARTING_POS;
    pos64 black_queens = BLACK_QUEEN_STARTING_POS;
    pos64 black_kings = BLACK_KING_STARTING_POS;
    
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
    pos64 white_pawns = WHITE_PAWN_STARTING_POS;
    pos64 white_bishops = WHITE_BISHOP_STARTING_POS;
    pos64 white_knights = WHITE_KNIGHT_STARTING_POS;
    pos64 white_rooks = WHITE_ROOK_STARTING_POS;
    pos64 white_queens = WHITE_QUEEN_STARTING_POS;
    pos64 white_kings = WHITE_KING_STARTING_POS;

    pos64 black_pawns = BLACk_PAWN_STARTING_POS;
    pos64 black_bishops = BLACK_BISHOP_STARTING_POS;
    pos64 black_knights = BLACK_KNIGHT_STARTING_POS;
    pos64 black_rooks = BLACK_ROOK_STARTING_POS;
    pos64 black_queens = BLACK_QUEEN_STARTING_POS;
    pos64 black_kings = BLACK_KING_STARTING_POS;

    short current_player = 0;

    pos64 *white_pawn_moves, *white_bishop_moves, *white_knight_moves, *white_rook_moves, *white_queen_moves, *white_king_moves, 
        *black_pawn_moves, *black_bishop_moves, *black_knight_moves, *black_rook_moves, *black_queen_moves, *black_king_moves;

    white_pawn_moves = new pos64[BOARDS_GENERATED];
    white_bishop_moves = new pos64[BOARDS_GENERATED];
    white_knight_moves = new pos64[BOARDS_GENERATED];
    white_rook_moves = new pos64[BOARDS_GENERATED];
    white_queen_moves = new pos64[BOARDS_GENERATED];
    white_king_moves = new pos64[BOARDS_GENERATED];
    black_pawn_moves = new pos64[BOARDS_GENERATED];
    black_bishop_moves = new pos64[BOARDS_GENERATED];
    black_knight_moves = new pos64[BOARDS_GENERATED];
    black_rook_moves = new pos64[BOARDS_GENERATED];
    black_queen_moves = new pos64[BOARDS_GENERATED];
    black_king_moves = new pos64[BOARDS_GENERATED];

    // when
    generate_moves(&white_pawns, &white_bishops, &white_knights, &white_rooks, &white_queens, &white_kings, 
                &black_pawns, &black_bishops, &black_knights, &black_rooks, &black_queens, &black_kings,
                white_pawn_moves, white_bishop_moves, white_knight_moves, white_rook_moves, white_queen_moves, white_king_moves, 
                black_pawn_moves, black_bishop_moves, black_knight_moves, black_rook_moves, black_queen_moves,  black_king_moves,
                current_player);
    int generated_moves_count = 0;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if ((black_king_moves[x] | white_king_moves[x]) == 0) break;
        generated_moves_count = x + 1;
    }

    // then 
    ASSERT_EQ(12, generated_moves_count);
}

TEST(SearchForBestMovesTest, StartPositionBestMoveSearchForWhite) {
    // given
    pos64 white_pawns = WHITE_PAWN_STARTING_POS;
    pos64 white_bishops = WHITE_BISHOP_STARTING_POS;
    pos64 white_knights = WHITE_KNIGHT_STARTING_POS;
    pos64 white_rooks = WHITE_ROOK_STARTING_POS;
    pos64 white_queens = WHITE_QUEEN_STARTING_POS;
    pos64 white_kings = WHITE_KING_STARTING_POS;

    pos64 black_pawns = BLACk_PAWN_STARTING_POS;
    pos64 black_bishops = BLACK_BISHOP_STARTING_POS;
    pos64 black_knights = BLACK_KNIGHT_STARTING_POS;
    pos64 black_rooks = BLACK_ROOK_STARTING_POS;
    pos64 black_queens = BLACK_QUEEN_STARTING_POS;
    pos64 black_kings = BLACK_KING_STARTING_POS;

    short current_player = 0;
    int move_num = 0;

    pos64 new_white_pawns = white_pawns;
    pos64 new_white_bishops = white_bishops;
    pos64 new_white_knights = white_knights;
    pos64 new_white_rooks = white_rooks;
    pos64 new_white_queens = white_queens;
    pos64 new_white_kings = white_kings;
    pos64 new_black_pawns = black_pawns;
    pos64 new_black_bishops = black_bishops;
    pos64 new_black_knights = black_knights;
    pos64 new_black_rooks = black_rooks;
    pos64 new_black_queens = black_queens;
    pos64 new_black_kings = black_kings;

    // when
    init();
    search(current_player, move_num,
           new_white_pawns, new_white_bishops, new_white_knights, new_white_rooks, new_white_queens, new_white_kings, 
           new_black_pawns, new_black_bishops, new_black_knights, new_black_rooks, new_black_queens, new_black_kings);
    terminate();
    pos64 current_pos_white = white_pawns | white_bishops | white_knights | white_rooks | white_queens | white_kings;
    pos64 new_pos_white = new_white_pawns | new_white_bishops | new_white_knights | new_white_rooks | new_white_queens | new_white_kings;
    pos64 current_pos_black = black_pawns | black_bishops | black_knights | black_rooks | black_queens | black_kings;
    pos64 new_pos_black = new_black_pawns | new_black_bishops | new_black_knights | new_black_rooks | new_black_queens | new_black_kings;
    
    // then
    ASSERT_EQ(new_pos_black, current_pos_black);
    ASSERT_NE(new_pos_white, current_pos_white);
}

TEST(SearchForBestMovesTest, StartPositionBestMoveSearchForBlack) {
    // given
    pos64 white_pawns = WHITE_PAWN_STARTING_POS;
    pos64 white_bishops = WHITE_BISHOP_STARTING_POS;
    pos64 white_knights = WHITE_KNIGHT_STARTING_POS;
    pos64 white_rooks = WHITE_ROOK_STARTING_POS;
    pos64 white_queens = WHITE_QUEEN_STARTING_POS;
    pos64 white_kings = WHITE_KING_STARTING_POS;

    pos64 black_pawns = BLACk_PAWN_STARTING_POS;
    pos64 black_bishops = BLACK_BISHOP_STARTING_POS;
    pos64 black_knights = BLACK_KNIGHT_STARTING_POS;
    pos64 black_rooks = BLACK_ROOK_STARTING_POS;
    pos64 black_queens = BLACK_QUEEN_STARTING_POS;
    pos64 black_kings = BLACK_KING_STARTING_POS;

    short current_player = 1;
    int move_num = 0;

    pos64 new_white_pawns = white_pawns;
    pos64 new_white_bishops = white_bishops;
    pos64 new_white_knights = white_knights;
    pos64 new_white_rooks = white_rooks;
    pos64 new_white_queens = white_queens;
    pos64 new_white_kings = white_kings;
    pos64 new_black_pawns = black_pawns;
    pos64 new_black_bishops = black_bishops;
    pos64 new_black_knights = black_knights;
    pos64 new_black_rooks = black_rooks;
    pos64 new_black_queens = black_queens;
    pos64 new_black_kings = black_kings;

    // when
    init();
    search(current_player, move_num,
           new_white_pawns, new_white_bishops, new_white_knights, new_white_rooks, new_white_queens, new_white_kings, 
           new_black_pawns, new_black_bishops, new_black_knights, new_black_rooks, new_black_queens, new_black_kings);
    terminate();
    pos64 current_pos_white = white_pawns | white_bishops | white_knights | white_rooks | white_queens | white_kings;
    pos64 new_pos_white = new_white_pawns | new_white_bishops | new_white_knights | new_white_rooks | new_white_queens | new_white_kings;
    pos64 current_pos_black = black_pawns | black_bishops | black_knights | black_rooks | black_queens | black_kings;
    pos64 new_pos_black = new_black_pawns | new_black_bishops | new_black_knights | new_black_rooks | new_black_queens | new_black_kings;
    
    // then
    ASSERT_EQ(new_pos_white, current_pos_white);
    ASSERT_NE(new_pos_black, current_pos_black);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}