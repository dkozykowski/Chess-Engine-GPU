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

        if (token == "exit" || token == "quit" ||  token == "stop") break;

        //else if (token == "go")         go(pos, is, states);
        //else if (token == "position")   position(pos, is, states);
        //else if (token == "ucinewgame") Search::clear();
        
        //else if (token == "bench")    bench(pos, is, states);
        else if (token == "d")        std::cout << pos << std::endl;
        //else if (token == "eval")     sync_cout << Eval::trace(pos) << sync_endl;
        else
            std::cout << "Unknown command: " << cmd << std::endl;
  } while (token != "quit"); // Command line args are one-shot

}

} // namespace UCI