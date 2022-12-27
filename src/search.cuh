#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "macros.cuh"

namespace SEARCH {

void init();

void terminate();

void findBestMove(const short& current_player, pos64* position);

} // namespace SEARCH

#endif // #ifndef SEARCH_H_INCLUDED 