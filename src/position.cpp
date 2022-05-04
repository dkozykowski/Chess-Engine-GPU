#include <algorithm>
#include <iterator>
#include <sstream>

#include "position.h"
#include "error.h"

Position::Position() {
    set_FEN(DEFAULT_FEN);
}

void Position::set_FEN(std::string FEN) {
    this->FEN = FEN;

    if (board) {
        for (int i = 0; i < 8; i++) {
            delete[] board[i];
        }
        delete[] board;
    }
    if ((board = new char*[8]) == nullptr)
        ERR("Allocation");
    for (int i = 0; i < 8; i++)
        if ((board[i] = new char[8]) == nullptr) 
            ERR("Allocation");
        else std::fill(board[i], board[i] + 8, ' ');
    
    int column = 0, row = 0;
    std::istringstream FENstream(FEN);
    char sign;
    while(FENstream >> sign) {
        if (sign == '/') continue;
        else if ('0' <= sign && sign <= '9') column += sign - '0';
        else board[column++][row] = sign;
        
        if (column == 8) row++, column = 0;
    }
}

Position::~Position() {
    for (int i = 0; i < 8; i++) {
        delete board[i];
    }
    delete[] board;
}

std::ostream& operator<<(std::ostream& os, const Position& pos) {
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            os << " ———";  
        }
        os << std::endl;
        os << '|';
        for (int column = 0; column < 8; column++) {
            os << ' ' << pos.board[column][row] << " |";  
        }
        os << std::endl;
    }
    for (int column = 0; column < 8; column++) {
        os << " ———";;  
    }
    return os;
}