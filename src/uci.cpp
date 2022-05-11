#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "uci.h"
#include "position.h"

namespace UCI {

void loop() {
    Position pos;
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
        else if (token == "ucinewgame") pos = Position();
        else if (token == "d")          std::cout << pos << std::endl;
        else if (token == "flip")       pos.flip();
        // else if (token == "go")         go(pos, is, states);
        // else if (token == "bench")      bench(pos, is, states);
        // else if (token == "eval")       sync_cout << Eval::trace(pos) << sync_endl;
        else
            std::cout << "Unknown command: " << cmd << std::endl;
  } while (true);

}

} // namespace UCI