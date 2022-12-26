#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <stdio.h>

#include <string>

#include "macros.cuh"

void printPosition(const pos64& whitePawns, const pos64& whiteBishops,
                   const pos64& whiteKnights, const pos64& whiteRooks,
                   const pos64& whiteQueens, const pos64& whiteKings,
                   const pos64& blackPawns, const pos64& blackBishops,
                   const pos64& blackKnights, const pos64& blackRooks,
                   const pos64& blackQueens, const pos64& blackKings);

void flipPosition(pos64& whitePawns, pos64& whiteBishops, pos64& whiteKnights,
                  pos64& whiteRooks, pos64& whiteQueens, pos64& whiteKings,
                  pos64& blackPawns, pos64& blackBishops, pos64& blackKnights,
                  pos64& blackRooks, pos64& blackQueens, pos64& blackKings);

void moveChess(const int& fromCol, const int& fromRow, const int& toCol,
               const int& toRow, short& currentPlayer, pos64& whitePawns,
               pos64& whiteBishops, pos64& whiteKnights, pos64& whiteRooks,
               pos64& whiteQueens, pos64& whiteKings, pos64& blackPawns,
               pos64& blackBishops, pos64& blackKnights, pos64& blackRooks,
               pos64& blackQueens, pos64& blackKings);

#endif