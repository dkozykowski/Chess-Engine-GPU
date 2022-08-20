#include "search.cuh"
#include "macros.cuh"
#include "evaluate.cuh"

// BASE BOARD IS DEPTH = 0
// NEXT LEVEL IS DEPTH = 1

int * subtree_sizes;
int * max_depth_to_store;

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

    int wsk = 0;
    if ((depth & 1) == 1) { // white moves
        for (wsk = 0; wsk < BOARDS_GENERATED; wsk++) {
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
    for (wsk = 0; wsk < BOARDS_GENERATED; wsk++) {
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



__device__ void gather_results(int* results, int* from, int depth) {
    if ((depth & 1) == 1) { // white moves, maximizes
        for (int i = 0; i < BOARDS_GENERATED; i++) {
            results[0] = max(results[0], from[i]);
        }
    } 
    else { // black moves, minimizes
        for (int i = 0; i < BOARDS_GENERATED; i++) {
            results[0] = min(results[0], from[i]);
        }
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
    results[0] = INF;
    depths[0] = 1;
    stack_states[0] = RIGHT;
    *stack_wsk = 0;
    *current_depth = 0;
    white_pawns_boards[0]   = white_pawns;
    white_bishops_boards[0]  = white_bishops;
    white_knights_boards[0]  = white_knights;
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

__global__ void initalize_subtree_sizes(int* subtree_sizes, int* max_depth_to_store) {
    subtree_sizes[MAX_DEPTH] = 1;
    for (int i = MAX_DEPTH - 1;  i >= 0; i--) {
        subtree_sizes[i] = subtree_sizes[i + 1] * BOARDS_GENERATED;
        *max_depth_to_store = i;
        if (subtree_sizes[i] + 1 >= MAX_BOARDS_SIMULTANEOUSLY / BOARDS_GENERATED) break;
    }
    subtree_sizes[MAX_DEPTH] = 0;

    //todo remove
    *max_depth_to_store = min(*max_depth_to_store, 3);
}

void init() {
    cudaMalloc(&subtree_sizes, sizeof(int) * MAX_DEPTH + 1);
    cudaMalloc(&max_depth_to_store, sizeof(int));
    initalize_subtree_sizes<<<1, 1>>>(subtree_sizes, max_depth_to_store);
    cudaDeviceSynchronize();
}

void terminate() {
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
                        //int offset,
                        int * current_depth,
                        int * subtree_sizes,
                        const int * max_depth_to_store,
                        int * stack_wsk) {
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

    int index = blockIdx.x * 1024 + threadIdx.x;
    if (*current_depth <= * max_depth_to_store) { // non full search
        if (index != 0) return;
        printf("Generating moves!\n");
        generate_moves(&white_pawns_boards[1],
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
                       depths[0] + 1);
        return;
    }
    // else { // fullsearch
    //     if (depth < MAX_DEPTH) {
    //     //     int parent_depth = depths[0];
    //     //     int layers_between = depth - parent_depth + 1;
    //     //     if (index < subtree_sizes[MAX_DEPTH - layers_between] 
    //     //         || index >= subtree_sizes[MAX_DEPTH - layers_between - 1]) 
    //     //             return;
    //     //     else {
    //     //         int local_index = index - subtree_sizes[MAX_DEPTH - layers_between];
    //     //         int layer_offset = subtree_sizes[MAX_DEPTH - layers_between - 1] - subtree_sizes[MAX_DEPTH - layers_between];
    //     //         generate_moves(&white_pawns_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &white_bishops_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &white_knights_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &white_rooks_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &white_queens_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &white_kings_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &black_pawns_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &black_bishops_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &black_knights_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &black_rooks_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &black_queens_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &black_kings_boards[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &results[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        &depths[local_index * BOARDS_GENERATED + layer_offset],
    //     //                        depths[0] + 1);
    //     //     }
    //     }
    //     else { // depth == MAX_DEPTH, evaluating layer
    //         // int local_index = index - subtree_sizes[MAX_DEPTH - layers_between]; // czy na pewno?
    //         // results[local_index] = evaluate_position(white_bishops_boards[local_index],
    //         //                                          white_knights_boards[local_index],
    //         //                                          white_rooks_boards[local_index],
    //         //                                          white_queens_boards[local_index],
    //         //                                          white_kings_boards[local_index],
    //         //                                          black_pawns_boards[local_index], 
    //         //                                          black_bishops_boards[local_index], 
    //         //                                          black_knights_boards[local_index],
    //         //                                          black_rooks_boards[local_index],
    //         //                                          black_queens_boards[local_index],
    //         //                                          black_kings_boards[local_index]);
    //     }
    // }
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
            int * max_depth_to_store) {

    if (depths[*stack_wsk] <= * max_depth_to_store) { // non full search
        if (stack_states[*stack_wsk] == LEFT) {
            printf("Zbieram wyniki z %d\n", *stack_wsk);
            gather_results(&results[*stack_wsk], &results[*stack_wsk], depths[*stack_wsk]);
            *stack_wsk -= 1;
        }
        else if (stack_states[*stack_wsk] == RIGHT) {
            printf("Ide dalej z %d\n", *stack_wsk);
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
        printf("Cofam z %d\n", *stack_wsk);
        *stack_wsk -= 1; // todo remove
    }
    // else {
    //     if (stack_direction[stack_wsk] == LEFT) {
            
    //         *stack_wsk--;
    //     }
    //     else if (stack_direction[stack_wsk] == RIGHT) {

    //         *stack_wsk--;
    //     }
    // }
    
}

void search(const int& current_player,
            const int& move_num,
            const pos64& white_pawns,
            const pos64& white_bishops,
            const pos64& white_knights,
            const pos64& white_rooks,
            const pos64& white_queens,
            const pos64& white_kings,
            const pos64& black_pawns,
            const pos64& black_bishops,
            const pos64& black_knights,
            const pos64& black_rooks,
            const pos64& black_queens,
            const pos64& black_kings) {

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

    h_search_ended = new bool;
    cudaMalloc(&white_pawns_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&white_bishops_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&white_knights_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&white_rooks_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&white_queens_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&white_kings_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&black_pawns_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&black_bishops_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&black_knights_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&black_rooks_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&black_queens_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&black_kings_boards, sizeof(pos64) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&results, sizeof(int) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&depths, sizeof(int) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&d_search_ended, sizeof(bool));
    cudaMalloc(&stack_states, sizeof(short) * MAX_BOARDS_IN_MEMORY);
    cudaMalloc(&stack_wsk, sizeof(int));
    cudaMalloc(&current_depth, sizeof(int));

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
                        //int offset,
                        current_depth,
                        subtree_sizes,
                        max_depth_to_store,
                        stack_wsk);
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
                max_depth_to_store);
        cudaDeviceSynchronize();

        cudaMemcpy(h_search_ended, d_search_ended, sizeof(bool), cudaMemcpyDeviceToHost);
    } while(!(*h_search_ended));

    delete h_search_ended;
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
}
