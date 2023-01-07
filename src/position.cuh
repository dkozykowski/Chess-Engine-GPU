#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <stdio.h>

#include <string>

#include "macros.cuh"

namespace POSITION {

void printPosition(pos64* position);

void moveChess(const int& fromCol, const int& fromRow, const int& toCol,
               const int& toRow, short& currentPlayer, pos64** position, 
               bool promote, bool toQueen);
}  // namespace POSITION

#endif  // #ifndef POSITION_H_INCLUDED