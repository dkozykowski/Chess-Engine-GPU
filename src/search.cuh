#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "macros.cuh"

namespace SEARCH {

void init();

void terminate();

long findBestMove(const short& current_player, pos64* position, int maxDevices = 8, int maxDepth = MAX_POSSIBLE_DEPTH);

}  // namespace SEARCH

#endif  // #ifndef SEARCH_H_INCLUDED