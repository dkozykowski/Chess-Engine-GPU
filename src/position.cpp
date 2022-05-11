#include <algorithm>
#include <iterator>
#include <sstream>

#include "position.h"
#include "macros.h"

Position::Position() {
    set_FEN(DEFAULT_FEN);
}

Position::Position(const Position& pos) {
    set_FEN(pos.FEN);
}

Position::~Position() {
    for (int i = 0; i < 8; i++) {
        delete board[i];
    }
    delete[] board;
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

void Position::flip() {
    std::string newFen = FEN;
    for (int i = 0; i <(int)newFen.size(); i++) {
        if ('a' <= newFen[i] && newFen[i] <= 'z') newFen[i] += 'A' - 'a';
        else if ('A' <= newFen[i] && newFen[i] <= 'Z') newFen[i] += 'a' - 'A';
        else if (newFen[i] == ' ') break;
    }
    set_FEN(newFen);
}

std::ostream& operator<<(std::ostream& os, const Position& pos) {
    os << "+---+---+---+---+---+---+---+---+\n";
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            os << "| " << pos.board[column][row] << ' ';  
        }
        os << "|\n+---+---+---+---+---+---+---+---+\n";
    }   
    return os;
}