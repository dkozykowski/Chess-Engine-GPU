#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "macros.cuh"
#include "uci.cuh"
#include "position.cuh"
#include "evaluate.cuh"
#include "search.cuh"

namespace UCI {

pos64 white_pawns;
pos64 white_bishops;
pos64 white_knights;
pos64 white_rooks;
pos64 white_queens;
pos64 white_kings;
pos64 black_pawns;
pos64 black_bishops;
pos64 black_knights;
pos64 black_rooks;
pos64 black_queens;
pos64 black_kings;

void newgame(int & current_player, int & move_num);
void move(std::istringstream & is, int & current_player, int & move_num);
void print_game(int current_player, int move_num);
void print_eval();
void go(int & current_player, int & move_num);

void loop() {
    int current_player, move_num;
    newgame(current_player, move_num);
    init();

    std::string token, cmd;

    do {
        if(!std::getline(std::cin, cmd)) break;

        std::istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> std::skipws >> token;

        if (token == "exit" || 
            token == "quit" ||  
            token == "stop" || 
            token == "q")               break;
        else if (token == "ucinewgame") newgame(current_player, move_num);
        else if (token == "d")          print_game(current_player, move_num);
        else if (token == "flip")       
            flip_position(white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                          black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
        else if (token == "move")       move(is, current_player, move_num);
        else if (token == "go")         go(current_player, move_num);
        // else if (token == "bench")      bench(pos, is, states);
        else if (token == "eval")       print_eval();
        else
            std::cout << "Unknown command: " << cmd << std::endl;
    } while (true);

    terminate();
}

void newgame(int & current_player, int & move_num) {
    white_pawns = WHITE_PAWN_STARTING_POS;
    white_bishops = WHITE_BISHOP_STARTING_POS;
    white_knights = WHITE_KNIGHT_STARTING_POS;
    white_rooks = WHITE_ROOK_STARTING_POS;
    white_queens = WHITE_QUEEN_STARTING_POS;
    white_kings = WHITE_KING_STARTING_POS;

    black_pawns = BLACk_PAWN_STARTING_POS;
    black_bishops = BLACK_BISHOP_STARTING_POS;
    black_knights = BLACK_KNIGHT_STARTING_POS;
    black_rooks = BLACK_ROOK_STARTING_POS;
    black_queens = BLACK_QUEEN_STARTING_POS;
    black_kings = BLACK_KING_STARTING_POS;

    current_player = WHITE;
    move_num = 0;
}

void move(std::istringstream & is, int & current_player, int & move_num) {
    std::string move_token;
    is >> std::skipws >> move_token;

    // validate
    if (move_token.length() != 4) {
        printf("Invalid move\n");
        return;
    }

    int from_col = move_token[0] >= 'a' ? move_token[0] - 'a' : move_token[0] - 'A';
    int from_row = '8' - move_token[1];
    int to_col = move_token[2] >= 'a' ? move_token[2] - 'a' : move_token[2] - 'A';
    int to_row = '8' - move_token[3];
    
    if (from_col < 0 || from_row < 0 || to_col < 0 || to_row < 0 ||
        8 <= from_col || 8 <= from_row || 8 <= to_col || 8 <= to_row) {
        printf("Invalid move\n");
        return;
    }

    move_chess(from_col, from_row, to_col, to_row, current_player,
               white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
               black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
    move_num++;
    current_player ^= 1;
}

void print_game(int current_player, int move_num) {
    printf("Move number %d\n", move_num);
    printf("Current player - %s\n", current_player == WHITE ? "White" : "Black");
    print_position(white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                   black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
}

__global__ void eval(int * result,
                pos64 white_pawns, pos64 white_bishops, pos64 white_knights, pos64 white_rooks, pos64 white_queens, pos64 white_kings,
                pos64 black_pawns, pos64 black_bishops, pos64 black_knights, pos64 black_rooks, pos64 black_queens, pos64 black_kings) {
    *result = evaluate_position(white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                          black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
}

void print_eval() {
    int * d_result, * h_result;
    h_result = new int;
    cudaMalloc(&d_result, sizeof(int));
    eval<<<1, 1>>>(d_result, white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                          black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Current evaluation from white side: %d\n", *h_result);
    delete h_result;
    cudaFree(d_result);
}

void go(int & current_player, int & move_num) {
    search(current_player, move_num,
           white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
           black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings);
}


} // namespace UCI