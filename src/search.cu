#include <algorithm>

#include "search.cuh"
#include "macros.cuh"
#include "evaluate.cuh"
#include "moves.cuh"
// BASE BOARD IS DEPTH = 0
// NEXT LEVEL IS DEPTH = 1

int * level_sizes;
int * subtree_sizes;
int * max_depth_to_store;

__device__ void gather_results(int* results, int* from, int depth, int * last_wsk) {
    if ((depth & 1) == 0) { // maximizing
        int result = -INF;
        for (int i = 0; i < BOARDS_GENERATED; i++) {
            if (from[i] != INF && from[i] != -INF && from[i] > result) {
                *last_wsk = i;
                result = from[i];
            }
        }
        results[0] = result;
    } 
    else { // minimizing
        int result = INF;
        for (int i = 0; i < BOARDS_GENERATED; i++) {
           if (from[i] != INF && from[i] != -INF && from[i] < result) {
                *last_wsk = i;
                result = from[i];
            }
        }
        results[0] = result;
    }
}  

__device__ void gather_small_results(int* results, int* from, int depth, int * last_wsk) {
    int i = 0;
    if ((depth & 1) == 0) { // maximizing
        int result = results[0] != INF && results[0] != -INF ? results[0] : -INF;
        
        if (from[i] != INF && from[i] != -INF && from[i] > result) {
            *last_wsk = i;
            result = from[i];
        }
        results[0] = result;
    } 
    else { // minimizing
       int result = results[0] != INF && results[0] != -INF ? results[0] : INF;
        if (from[i] != INF && from[i] != -INF && from[i] < result) {
            *last_wsk = i;
            result = from[i];
        }
        results[0] = result;
    }
}  

__global__ void init_searching(pos64 * white_pawns_boards,
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
                    const pos64 white_pawns,
                    const pos64 white_bishops,
                    const pos64 white_knights,
                    const pos64 white_rooks,
                    const pos64 white_queens,
                    const pos64 white_kings,
                    const pos64 black_pawns,
                    const pos64 black_bishops,
                    const pos64 black_knights,
                    const pos64 black_rooks,
                    const pos64 black_queens,
                    const pos64 black_kings,
                    int * results,
                    int * depths,
                    short* stack_states,
                    int* stack_wsk,
                    int* current_depth) {
    depths[0] = 0;
    stack_states[0] = RIGHT;
    *stack_wsk = 0;
    *current_depth = 0;
    white_pawns_boards[0]   = white_pawns;
    white_bishops_boards[0] = white_bishops;
    white_knights_boards[0] = white_knights;
    white_rooks_boards[0]   = white_rooks;
    white_queens_boards[0]  = white_queens;
    white_kings_boards[0]   = white_kings;
    black_pawns_boards[0]   = black_pawns;
    black_bishops_boards[0] = black_bishops;
    black_knights_boards[0] = black_knights;
    black_rooks_boards[0]   = black_rooks;
    black_queens_boards[0]  = black_queens;
    black_kings_boards[0]   = black_kings;
}

__global__ void init_sizes_tables(int* level_sizes, int * subtree_sizes, int* max_depth_to_store) {
    level_sizes[MAX_DEPTH] = 1;
    subtree_sizes[MAX_DEPTH] = 1;
    subtree_sizes[MAX_DEPTH + 1] = 0;
    bool still_fine = true;
    for (int i = MAX_DEPTH - 1;  i >= 0; i--) {
        level_sizes[i] = level_sizes[i + 1] * BOARDS_GENERATED;
        subtree_sizes[i] = subtree_sizes[i + 1] * BOARDS_GENERATED + 1;
        if (level_sizes[i] * BOARDS_GENERATED >= MAX_BOARDS_SIMULTANEOUSLY) {
            still_fine = false;
        }
        if (still_fine) {
            *max_depth_to_store = i;
        }
    }
}

void init() {
    CHECK_ALLOC(cudaMalloc(&level_sizes, sizeof(int) * MAX_DEPTH + 1));
    CHECK_ALLOC(cudaMalloc(&subtree_sizes, sizeof(int) * MAX_DEPTH + 2));
    CHECK_ALLOC(cudaMalloc(&max_depth_to_store, sizeof(int)));
    init_sizes_tables<<<1, 1>>>(level_sizes, subtree_sizes, max_depth_to_store);
    cudaDeviceSynchronize();
}

void terminate() {
    cudaFree(level_sizes);
    cudaFree(subtree_sizes);
    cudaFree(max_depth_to_store);
}

__global__ void do_searching(pos64 * white_pawns_boards,
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
                        int * current_depth,
                        int * level_sizes,
                        int * subtree_sizes,
                        const int * max_depth_to_store,
                        int * stack_wsk,
                        int * last_wsk,
                        short current_player) {
    white_pawns_boards = &white_pawns_boards[*stack_wsk]; 
    white_bishops_boards = &white_bishops_boards[*stack_wsk];
    white_knights_boards = &white_knights_boards[*stack_wsk];
    white_rooks_boards = &white_rooks_boards[*stack_wsk];
    white_queens_boards = &white_queens_boards[*stack_wsk];
    white_kings_boards = &white_kings_boards[*stack_wsk];
    black_pawns_boards = &black_pawns_boards[*stack_wsk];
    black_bishops_boards = &black_bishops_boards[*stack_wsk];
    black_knights_boards = &black_knights_boards[*stack_wsk];
    black_rooks_boards = &black_rooks_boards[*stack_wsk];
    black_queens_boards = &black_queens_boards[*stack_wsk];
    black_kings_boards = &black_kings_boards[*stack_wsk];
    stack_states = &stack_states[*stack_wsk];
    results = &results[*stack_wsk];
    depths = &depths[*stack_wsk];
    current_player = (current_player ^ (*current_depth & 1));

    int index = blockIdx.x * 1024 + threadIdx.x;
    if (*current_depth < * max_depth_to_store) { // non full search
        if (index != 0) return;
        DBG(printf("Generating moves!\n"));
        generate_moves(&white_pawns_boards[0],
                       &white_bishops_boards[0],
                       &white_knights_boards[0],
                       &white_rooks_boards[0],
                       &white_queens_boards[0],
                       &white_kings_boards[0],
                       &black_pawns_boards[0],
                       &black_bishops_boards[0],
                       &black_knights_boards[0],
                       &black_rooks_boards[0],
                       &black_queens_boards[0],
                       &black_kings_boards[0],
                       &white_pawns_boards[1],
                       &white_bishops_boards[1],
                       &white_knights_boards[1],
                       &white_rooks_boards[1],
                       &white_queens_boards[1],
                       &white_kings_boards[1],
                       &black_pawns_boards[1],
                       &black_bishops_boards[1],
                       &black_knights_boards[1],
                       &black_rooks_boards[1],
                       &black_queens_boards[1],
                       &black_kings_boards[1],
                       &results[1],
                       &depths[1],
                       &stack_states[1],
                       depths[0] + 1,
                       current_player);
    }
    else { // fullsearch
        int depth_difference = *current_depth - depths[0];
        DBG2(printf("Ww %d akceptuje %d\n", *stack_wsk, level_sizes[MAX_DEPTH - depth_difference - 1]));
        if (index >= level_sizes[MAX_DEPTH - depth_difference - 1]) {
            return;
        }

        int offset = subtree_sizes[MAX_DEPTH - depth_difference];
        int global_index = offset + index;
        DBG2(printf("%d %d %d\n", depth_difference, offset, global_index + *stack_wsk));

        if (*current_depth == MAX_DEPTH) { // evaluating layer    
            DBG2(printf("Ewaluuje %d\n", global_index + *stack_wsk));
            if ((white_kings_boards[global_index] | black_kings_boards[global_index]) == 0) {
                results[global_index] = INF;
            }
            else {
                results[global_index] = evaluate_position(white_pawns_boards[global_index],
                                                        white_bishops_boards[global_index],
                                                        white_knights_boards[global_index],
                                                        white_rooks_boards[global_index],
                                                        white_queens_boards[global_index],
                                                        white_kings_boards[global_index],
                                                        black_pawns_boards[global_index], 
                                                        black_bishops_boards[global_index], 
                                                        black_knights_boards[global_index],
                                                        black_rooks_boards[global_index],
                                                        black_queens_boards[global_index],
                                                        black_kings_boards[global_index]);
            }
        }
        else {
            int sons_offset = subtree_sizes[MAX_DEPTH - depth_difference - 1];
            int sons_index = sons_offset + index * BOARDS_GENERATED;
            if (stack_states[0] == LEFT) {
                DBG2(printf("Zbieram wyniki z %d {2} od pozycji %d\n", global_index + *stack_wsk, sons_index + *stack_wsk));
                gather_results(&results[global_index], &results[sons_index], depths[global_index], &last_wsk[global_index]);
            }
            else {
                DBG2(printf("Synowie %d od %d i im generuje\n", global_index + *stack_wsk, sons_index + *stack_wsk));
                generate_moves(&white_pawns_boards[global_index],
                                &white_bishops_boards[global_index],
                                &white_knights_boards[global_index],
                                &white_rooks_boards[global_index],
                                &white_queens_boards[global_index],
                                &white_kings_boards[global_index],
                                &black_pawns_boards[global_index],
                                &black_bishops_boards[global_index],
                                &black_knights_boards[global_index],
                                &black_rooks_boards[global_index],
                                &black_queens_boards[global_index],
                                &black_kings_boards[global_index],
                                &white_pawns_boards[sons_index],
                                &white_bishops_boards[sons_index],
                                &white_knights_boards[sons_index],
                                &white_rooks_boards[sons_index],
                                &white_queens_boards[sons_index],
                                &white_kings_boards[sons_index],
                                &black_pawns_boards[sons_index],
                                &black_bishops_boards[sons_index],
                                &black_knights_boards[sons_index],
                                &black_rooks_boards[sons_index],
                                &black_queens_boards[sons_index],
                                &black_kings_boards[sons_index],
                                &results[sons_index],
                                &depths[sons_index],
                                &stack_states[sons_index],
                                0,
                                current_player);
            }
        }    
    }
}

__global__ void search_main(pos64 * white_pawns_boards,
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
            int * stack_wsk, 
            short * stack_states,
            int * current_depth,
            bool * search_ended,
            int * max_depth_to_store,
            int * level_sizes,
            int * subtree_sizes,
            int * last_wsk,
            int * fathers) {

    if (depths[*stack_wsk] < * max_depth_to_store) { // non full search
        if (stack_states[*stack_wsk] == LEFT) {
            DBG(printf("Zbieram wyniki z %d\n", *stack_wsk));
            gather_results(&results[*stack_wsk], &results[*stack_wsk + 1], depths[*stack_wsk], &last_wsk[*stack_wsk]);
            DBG(printf("Aktualizuje wynik ojcu %d\n", fathers[*stack_wsk]));
            gather_small_results(&results[fathers[*stack_wsk]], &results[*stack_wsk], depthsfathers[*stack_wsk], &last_wsk[fathers[*stack_wsk]]);
            *stack_wsk -= 1;
        }
        else if (stack_states[*stack_wsk] == RIGHT) {
            DBG(printf("Ide dalej z %d do %d na glebokosc %d\n", *stack_wsk, *stack_wsk + BOARDS_GENERATED, depths[*stack_wsk + BOARDS_GENERATED]));
            stack_states[*stack_wsk] = LEFT;
            *stack_wsk += BOARDS_GENERATED;
        }
        if (*stack_wsk == -1) {
            *search_ended = true;
            return;
        }
        *current_depth = depths[*stack_wsk];
    }
    else {
        if (stack_states[*stack_wsk] == RIGHT) {
            if (*current_depth == MAX_DEPTH) {
                stack_states[*stack_wsk] = LEFT;
                DBG(printf("Cofam sie z poziomami z %d do %d\n", *current_depth, *current_depth - 1));
                *current_depth -= 1;
            else {
                DBG(printf("Rozszerzam w prawo z %d. ", *stack_wsk));
                DBG(printf("Ide dalej z poziomami z %d do %d\n", *current_depth, *current_depth + 1));
                *current_depth += 1;
            }
        }
        else if (stack_states[*stack_wsk] == LEFT) {
            if (*current_depth == depths[*stack_wsk]) {
                DBG(printf("Zbieram wyniki z %d\n", *stack_wsk));
                gather_results(&results[*stack_wsk], &results[*stack_wsk + 1], depths[*stack_wsk], &last_wsk[*stack_wsk]);
                *stack_wsk -= 1;
            }
            else {
                DBG(printf("Cofam sie z poziomami z %d do %d\n", *current_depth, *current_depth - 1));
                *current_depth -= 1;
            }
        }
    }
}

void search(const short& current_player,
            const int& move_num,
            pos64& white_pawns,
            pos64& white_bishops,
            pos64& white_knights,
            pos64& white_rooks,
            pos64& white_queens,
            pos64& white_kings,
            pos64& black_pawns,
            pos64& black_bishops,
            pos64& black_knights,
            pos64& black_rooks,
            pos64& black_queens,
            pos64& black_kings) {

    pos64*  white_pawns_boards;
    pos64*  white_bishops_boards;
    pos64*  white_knights_boards;
    pos64*  white_rooks_boards;
    pos64*  white_queens_boards;
    pos64*  white_kings_boards;
    pos64*  black_pawns_boards;
    pos64*  black_bishops_boards;
    pos64*  black_knights_boards;
    pos64*  black_rooks_boards;
    pos64*  black_queens_boards;
    pos64*  black_kings_boards;
    int* results;
    int* depths; 
    bool* d_search_ended;
    bool* h_search_ended;
    short* stack_states; 
    int* stack_wsk;
    int* current_depth;
    int * last_wsk;

    h_search_ended = new bool;
    CHECK_ALLOC(cudaMalloc(&white_pawns_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));  
    CHECK_ALLOC(cudaMalloc(&white_bishops_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&white_knights_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&white_rooks_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&white_queens_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&white_kings_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&black_pawns_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&black_bishops_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&black_knights_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&black_rooks_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&black_queens_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&black_kings_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&results, sizeof(int) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&depths, sizeof(int) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&d_search_ended, sizeof(bool)));
    CHECK_ALLOC(cudaMalloc(&stack_states, sizeof(short) * MAX_BOARDS_IN_MEMORY));
    CHECK_ALLOC(cudaMalloc(&stack_wsk, sizeof(int)));
    CHECK_ALLOC(cudaMalloc(&current_depth, sizeof(int)));
    CHECK_ALLOC(cudaMalloc(&last_wsk, sizeof(int)));


    init_searching<<<1, 1>>>(white_pawns_boards,
                    white_bishops_boards,
                    white_knights_boards,
                    white_rooks_boards,
                    white_queens_boards,
                    white_kings_boards,
                    black_pawns_boards,
                    black_bishops_boards,
                    black_knights_boards,
                    black_rooks_boards,
                    black_queens_boards,
                    black_kings_boards,
                    white_pawns,
                    white_bishops,
                    white_knights,
                    white_rooks,
                    white_queens,
                    white_kings,
                    black_pawns,
                    black_bishops,
                    black_knights,
                    black_rooks,
                    black_queens,
                    black_kings,
                    results,
                    depths,
                    stack_states,
                    stack_wsk,
                    current_depth);

    do {

        do_searching<<<BLOCKS, THREADS>>>(white_pawns_boards,
                        white_bishops_boards,
                        white_knights_boards,
                        white_rooks_boards,
                        white_queens_boards,
                        white_kings_boards,
                        black_pawns_boards,
                        black_bishops_boards,
                        black_knights_boards,
                        black_rooks_boards,
                        black_queens_boards,
                        black_kings_boards,
                        results,
                        depths,
                        stack_states,
                        current_depth,
                        level_sizes,
                        subtree_sizes,
                        max_depth_to_store,
                        stack_wsk,
                        last_wsk,
                        current_player);
        cudaDeviceSynchronize();

        search_main<<<1, 1>>>(white_pawns_boards,
                white_bishops_boards,
                white_knights_boards,
                white_rooks_boards,
                white_queens_boards,
                white_kings_boards,
                black_pawns_boards,
                black_bishops_boards,
                black_knights_boards,
                black_rooks_boards,
                black_queens_boards,
                black_kings_boards,
                results,
                depths,
                stack_wsk,
                stack_states,
                current_depth,
                d_search_ended,
                max_depth_to_store,
                level_sizes,
                subtree_sizes,
                last_wsk);
        cudaDeviceSynchronize();

        cudaMemcpy(h_search_ended, d_search_ended, sizeof(bool), cudaMemcpyDeviceToHost);
    } while(!(*h_search_ended));

    int * wsk = new int;
    cudaMemcpy(wsk, last_wsk, sizeof(int), cudaMemcpyDeviceToHost);
    *wsk += 1;
    cudaMemcpy(&white_pawns, &white_pawns_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&white_bishops, &white_bishops_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&white_knights, &white_knights_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&white_rooks, &white_rooks_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&white_queens, &white_queens_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&white_kings, &white_kings_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&black_pawns, &black_pawns_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&black_bishops, &black_bishops_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&black_knights, &black_knights_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&black_rooks, &black_rooks_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&black_queens, &black_queens_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&black_kings, &black_kings_boards[*wsk], sizeof(pos64), cudaMemcpyDeviceToHost);

    delete h_search_ended;
    delete wsk;
    cudaFree(white_pawns_boards);
    cudaFree(white_bishops_boards);
    cudaFree(white_knights_boards);
    cudaFree(white_rooks_boards);
    cudaFree(white_queens_boards);
    cudaFree(white_kings_boards);
    cudaFree(black_pawns_boards);
    cudaFree(black_bishops_boards);
    cudaFree(black_knights_boards);
    cudaFree(black_rooks_boards);
    cudaFree(black_queens_boards);
    cudaFree(black_kings_boards);
    cudaFree(results);
    cudaFree(depths);
    cudaFree(d_search_ended);
    cudaFree(stack_states);
    cudaFree(stack_wsk);
    cudaFree(current_depth);
    cudaFree(last_wsk);
}
