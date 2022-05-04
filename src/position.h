#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#define DEFAULT_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#include <string>
#include <iostream>

class Position {
private:
    std::string FEN; // Forsyth-Edwards Notation
    char** board = nullptr;
public: 
    Position();
    ~Position();

    void set_FEN(std::string FEN);

    friend std::ostream& operator<<(std::ostream& os, const Position& pos);
};

#endif