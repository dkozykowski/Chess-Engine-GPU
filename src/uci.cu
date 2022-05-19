#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "macros.cuh"
#include "uci.cuh"
#include "position.cuh"
#include "evaluate.cuh"

namespace UCI {

void loop() {
    char* board = (char*)malloc(sizeof(char) * 64);
    if (!board) ERR("Malloc");

    init_tables();

    memcpy(board, START_POS, 64 * sizeof(char));
    int current_player = WHITE;
    int move_num = 0;

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
        else if (token == "ucinewgame") {
            memcpy(board, START_POS, 64 * sizeof(char));
            current_player = WHITE;
            move_num = 0;
        }
        else if (token == "d")          print_position(board);
        else if (token == "flip")       flip_position(board);
        // else if (token == "go")         go(pos, is, states);
        // else if (token == "bench")      bench(pos, is, states);
        else if (token == "eval")       printf("Current evaluation: %d\n", evaluate_position(board));
        else
            std::cout << "Unknown command: " << cmd << std::endl;
  } while (true);

  free(board);
}

} // namespace UCI