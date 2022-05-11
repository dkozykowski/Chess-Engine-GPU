#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <string>
#include <iostream>

class Position {
private:
    std::string FEN; // Forsyth-Edwards Notation
    char** board = nullptr;
public: 
    Position();
    Position(const Position& pos);
    ~Position();

    void set_FEN(std::string FEN);
    void do_move(/* params */);

    // for debugging
    void flip();
    friend std::ostream& operator<<(std::ostream& os, const Position& pos);
};

#endif