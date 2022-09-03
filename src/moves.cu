#include "moves.cuh"

__device__ void generate_moves(pos64 * white_pawns_boards,
                    pos64 * white_bishops_boards,
                    pos64 * white_knights_boards,
                    pos64 * white_rooks_boards,
                    pos64 * white_queens_boards,
                    pos64 * white_kings_boards,
                    pos64 * black_pawns_boards,
                    pos64 * black_bishops_boards,
                    pos64 * black_knights_boards,
                    pos64 * black_rooks_boards,
                    pos64 * black_queens_boards,
                    pos64 * black_kings_boards,
                    int * results,
                    int * depths,
                    short * stack_states,
                    int depth) {
    // zakladamy, ze korzeniem drzewa, czyli graczem dla którego szukamu ruchu, jest gracz CZARNY
    // zakladamy, ze korzen ma numer 0

    // pamietaj zeby pozostale miejsca nadpisac zerami!!!!!!!!!!!!!!!!!!

    if ((depth & 1) == 1) { // white moves
        for (int wsk = 0; wsk < BOARDS_GENERATED; wsk++) {
            white_pawns_boards[wsk] = white_pawns_boards[0];
            white_bishops_boards[wsk] = white_bishops_boards[0];
            white_knights_boards[wsk] = white_knights_boards[0];
            white_rooks_boards[wsk] = white_rooks_boards[0];
            white_queens_boards[wsk] = white_queens_boards[0];
            white_kings_boards[wsk] = white_kings_boards[0];
            depths[wsk] = depth;
            stack_states[wsk] = RIGHT;
            results[wsk] = -INF; // white maximizes
        }
    } 
    else { // black moves
        for (int wsk = 0; wsk < BOARDS_GENERATED; wsk++) {
            black_pawns_boards[wsk] = black_pawns_boards[0];
            black_bishops_boards[wsk] = black_bishops_boards[0];
            black_knights_boards[wsk] = black_knights_boards[0];
            black_rooks_boards[wsk] = black_rooks_boards[0];
            black_queens_boards[wsk] = black_queens_boards[0];
            black_kings_boards[wsk] = black_kings_boards[0];
            depths[wsk] = depth;
            stack_states[wsk] = RIGHT;
            results[wsk] = INF; // black minimizes
        }
    }
}