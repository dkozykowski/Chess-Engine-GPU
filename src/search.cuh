#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "macros.cuh"

void init();

void terminate();

void search(const short& current_player, pos64* position);

#endif