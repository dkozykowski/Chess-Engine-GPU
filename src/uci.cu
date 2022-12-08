#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "macros.cuh"
#include "uci.cuh"
#include "position.cuh"
#include "evaluate.cuh"
#include "search.cuh"
#include "moves.cuh"

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

void newgame(short & current_player, int & move_num);
void move(std::istringstream & is, short & current_player, int & move_num);
void print_game(short current_player, int move_num);
void print_eval();
void go(short & current_player, int & move_num);
void print_moves(pos64 white_pawns, pos64 white_bishops, pos64 white_knights, pos64 white_rooks, pos64 white_queens, pos64 white_kings,
                pos64 black_pawns, pos64 black_bishops, pos64 black_knights, pos64 black_rooks, pos64 black_queens, pos64 black_kings, short current_player);
std::string get_move_string(pos64 current_pos, pos64 new_pos);

int _log2(pos64 x);

void loop() {
    short current_player;
    int move_num;
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
        else if (token == "moves")      print_moves(white_pawns, white_bishops, white_knights, white_rooks, white_queens, white_kings, 
                          black_pawns, black_bishops, black_knights, black_rooks, black_queens, black_kings, current_player);
        else
            std::cout << "Unknown command: " << cmd << std::endl;
    } while (true);

    terminate();
}

void newgame(short & current_player, int & move_num) {
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

void move(std::istringstream & is, short & current_player, int & move_num) {
    std::string move_token;
    is >> std::skipws >> move_token;

    // validate
    if (move_token.length() != 4) {
        printf("Invalid move\n");
        return;
    }

    int from_col = move_token[0] >= 'a' ? move_token[0] - 'a' : move_token[0] - 'A';
    int from_row = move_token[1] - '1';
    int to_col = move_token[2] >= 'a' ? move_token[2] - 'a' : move_token[2] - 'A';
    int to_row = move_token[3] - '1';
    
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

void print_game(short current_player, int move_num) {
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

void go(short & current_player, int & move_num) {

    pos64 * position = new pos64[12];
    position[WHITE_PAWN_OFFSET] = white_pawns;
    position[WHITE_BISHOP_OFFSET] = white_bishops;
    position[WHITE_KNIGHT_OFFSET] = white_knights;
    position[WHITE_ROOK_OFFSET] = white_rooks;
    position[WHITE_QUEEN_OFFSET] = white_queens;
    position[WHITE_KING_OFFSET] = white_kings;
    position[BLACK_PAWN_OFFSET] = black_pawns;
    position[BLACK_BISHOP_OFFSET] = black_bishops;
    position[BLACK_KNIGHT_OFFSET] = black_knights;
    position[BLACK_ROOK_OFFSET] = black_rooks;
    position[BLACK_QUEEN_OFFSET] = black_queens;
    position[BLACK_KING_OFFSET] = black_kings;
    
    search(current_player, position);

    pos64 new_white_pawns = position[WHITE_PAWN_OFFSET];
    pos64 new_white_bishops = position[WHITE_BISHOP_OFFSET];
    pos64 new_white_knights = position[WHITE_KNIGHT_OFFSET];
    pos64 new_white_rooks = position[WHITE_ROOK_OFFSET];
    pos64 new_white_queens = position[WHITE_QUEEN_OFFSET];
    pos64 new_white_kings = position[WHITE_KING_OFFSET];
    pos64 new_black_pawns = position[BLACK_PAWN_OFFSET];
    pos64 new_black_bishops = position[BLACK_BISHOP_OFFSET];
    pos64 new_black_knights = position[BLACK_KNIGHT_OFFSET];
    pos64 new_black_rooks = position[BLACK_ROOK_OFFSET];
    pos64 new_black_queens = position[BLACK_QUEEN_OFFSET];
    pos64 new_black_kings = position[BLACK_KING_OFFSET];

    DBG2(print_position(
        new_white_pawns, new_white_bishops, new_white_knights, new_white_rooks, new_white_queens, new_white_kings, 
        new_black_pawns, new_black_bishops, new_black_knights, new_black_rooks, new_black_queens, new_black_kings
    ));

    if (current_player == WHITE) {
        pos64 current_pos = white_pawns | white_bishops | white_knights | white_rooks | white_queens | white_kings;
        pos64 new_pos = new_white_pawns | new_white_bishops | new_white_knights | new_white_rooks | new_white_queens | new_white_kings;
        std::cout << get_move_string(current_pos, new_pos) << "\n";
    }
    else if (current_player == BLACK) {
        pos64 current_pos = black_pawns | black_bishops | black_knights | black_rooks | black_queens | black_kings;
        pos64 new_pos = new_black_pawns | new_black_bishops | new_black_knights | new_black_rooks | new_black_queens | new_black_kings;
        std::cout << get_move_string(current_pos, new_pos) << "\n";
    }
}

std::string get_move_string(pos64 current_pos, pos64 new_pos) {
    pos64 diff = current_pos ^ new_pos;
    pos64 from = current_pos & diff;
    pos64 to = new_pos & diff;
    int from_pos = _log2(from);
    int to_pos = _log2(to);
    std::string result = "____";
    result[0] = from_pos % 8 + 'a';
    result[1] = from_pos / 8 + '1';
    result[2] = to_pos % 8 + 'a';
    result[3] = to_pos / 8 + '1';
    return result;
}

int _log2(pos64 x) { // asserting x is a power of two
    for (int i = 0; i < x; i++) {
        if ((x & (((pos64)1) << i)) != 0) {
            return i;
        }
    }
    return 0;
}

void print_moves(pos64 white_pawns, pos64 white_bishops, pos64 white_knights, pos64 white_rooks, pos64 white_queens, pos64 white_kings,
                pos64 black_pawns, pos64 black_bishops, pos64 black_knights, pos64 black_rooks, pos64 black_queens, pos64 black_kings, short current_player) 
{
    pos64 *position = new pos64[12];
    pos64 *generatedBoards = new pos64[BOARDS_GENERATED * BOARD_SIZE];

    position[WHITE_PAWN_OFFSET] = white_pawns;
    position[WHITE_BISHOP_OFFSET] = white_bishops;
    position[WHITE_KNIGHT_OFFSET] = white_knights;
    position[WHITE_ROOK_OFFSET] = white_rooks;
    position[WHITE_QUEEN_OFFSET] = white_queens;
    position[WHITE_KING_OFFSET] = white_kings;
    position[BLACK_PAWN_OFFSET] = black_pawns;
    position[BLACK_BISHOP_OFFSET] = black_bishops;
    position[BLACK_KNIGHT_OFFSET] = black_knights;
    position[BLACK_ROOK_OFFSET] = black_rooks;
    position[BLACK_QUEEN_OFFSET] = black_queens;
    position[BLACK_KING_OFFSET] = black_kings;

    generate_moves(position, generatedBoards, current_player == WHITE);
    std::string any;
    for(int x = 0; x < BOARDS_GENERATED; x++)
    {
        if (((generatedBoards + (x * BOARD_SIZE))[BLACK_KING_OFFSET] | (generatedBoards + (x * BOARD_SIZE))[WHITE_KING_OFFSET]) == 0) break;
        print_position((generatedBoards + (x * BOARD_SIZE))[WHITE_PAWN_OFFSET], (generatedBoards + (x * BOARD_SIZE))[WHITE_BISHOP_OFFSET], (generatedBoards + (x * BOARD_SIZE))[WHITE_KNIGHT_OFFSET],
         (generatedBoards + (x * BOARD_SIZE))[WHITE_ROOK_OFFSET], (generatedBoards + (x * BOARD_SIZE))[WHITE_QUEEN_OFFSET], (generatedBoards + (x * BOARD_SIZE))[WHITE_KING_OFFSET], 
        (generatedBoards + (x * BOARD_SIZE))[BLACK_PAWN_OFFSET], (generatedBoards + (x * BOARD_SIZE))[BLACK_BISHOP_OFFSET], (generatedBoards + (x * BOARD_SIZE))[BLACK_KNIGHT_OFFSET],
         (generatedBoards + (x * BOARD_SIZE))[BLACK_ROOK_OFFSET], (generatedBoards + (x * BOARD_SIZE))[BLACK_QUEEN_OFFSET], (generatedBoards + (x * BOARD_SIZE))[BLACK_KING_OFFSET]);
        
        std::getline(std::cin, any);
        if(any == "q")
            break;
    }

    free(position);
    free(generatedBoards);
}
} // namespace UCI