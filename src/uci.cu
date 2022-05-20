#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "macros.cuh"
#include "uci.cuh"
#include "position.cuh"
#include "evaluate.cuh"
#include "moves.cuh"

namespace UCI {

void newgame(char * board, int & current_player, int & move_num);
void move(char * board, std::istringstream & is, int current_player);
void print_game(char * board, int current_player, int move_num);
void print_eval(char * board);

void loop() {
    int current_player, move_num;
    char* board = (char*)malloc(sizeof(char) * 64);
    if (!board) ERR("Malloc");

    init_tables();
    newgame(board, current_player, move_num);

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
        else if (token == "ucinewgame") newgame(board, current_player, move_num);
        else if (token == "d")          print_game(board, current_player, move_num);
        else if (token == "flip")       flip_position(board);
        else if (token == "move")       move(board, is, current_player);
        // else if (token == "go")         go(pos, is, states);
        // else if (token == "bench")      bench(pos, is, states);
        else if (token == "eval")       print_eval(board);
        else
            std::cout << "Unknown command: " << cmd << std::endl;
  } while (true);

  free(board);
}

void newgame(char * board, int & current_player, int & move_num) {
    memcpy(board, START_POS, 64 * sizeof(char));
    current_player = WHITE;
    move_num = 0;
}

void move(char * board, std::istringstream & is, int current_player) {
    std::string move_token;
    is >> std::skipws >> move_token;
    if (move_token[0] < 'A') move_token[0] += 'A' - 'a';
    if (move_token[2] < 'A') move_token[2] += 'A' - 'a';

    make_move(board, 
              current_player, 
              move_token[0] - 'A',
              8 - (move_token[1] - '0'),
              move_token[2] - 'A',
              8 - (move_token[3] - '0'));
}

void print_game(char * board, int current_player, int move_num) {
    printf("Move number %d\n", move_num);
    printf("Current player - %s\n", current_player == WHITE ? "White" : "Black");
    print_position(board);
}

void print_eval(char * board) {
    printf("Current evaluation from white side: %d\n", evaluate_position(board));
}

} // namespace UCI