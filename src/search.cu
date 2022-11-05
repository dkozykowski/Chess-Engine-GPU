#include <algorithm>

#include "search.cuh"
#include "macros.cuh"
#include "evaluate.cuh"
#include "moves.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int * h_level_sizes;
int * d_level_sizes;
int * h_subtree_sizes;
int * d_subtree_sizes;
int * last;

__device__ void gather_results(int* results_to, int* results_from, bool maximize, int * last) {
    int result;
    if (maximize) { // maximizing
        result = -INF;
        for (int i = 0; i < BOARDS_GENERATED; i++) {
            if (results_from[i] != INF && results_from[i] != -INF && results_from[i] > result) {
                result = results_from[i];
                *last = i;
            }
        }
    } 
    else { // minimizing
        result = INF;
        for (int i = 0; i < BOARDS_GENERATED; i++) {
           if (results_from[i] != INF && results_from[i] != -INF && results_from[i] < result) {
                result = results_from[i];
                *last = i;
            }
        }
    }
    DBG(printf("Zebralem wyniki i mam %d\n", result));
    *results_to = result;
}  

__device__ __host__ void _init_sizes_tables(int* level_sizes, int * subtree_sizes) {
    level_sizes[0] = 1;
    subtree_sizes[0] = 1;
    for (int i = 1;  i <= MAX_DEPTH; i++) {
        level_sizes[i] = level_sizes[i - 1] * BOARDS_GENERATED;
        subtree_sizes[i] = level_sizes[i - 1] * BOARDS_GENERATED + subtree_sizes[i - 1];
    }
}

__global__ void init_sizes_tables(int* level_sizes, int * subtree_sizes) {
    _init_sizes_tables(level_sizes, subtree_sizes);   
}

void init() {
    h_level_sizes = new int[MAX_DEPTH + 1];
    h_subtree_sizes = new int[MAX_DEPTH + 2];
    CHECK_ALLOC(cudaMalloc(&d_level_sizes, sizeof(int) * MAX_DEPTH + 1));
    CHECK_ALLOC(cudaMalloc(&d_subtree_sizes, sizeof(int) * MAX_DEPTH + 2));
    CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));

    init_sizes_tables<<<1, 1>>>(d_level_sizes, d_subtree_sizes);
    _init_sizes_tables(h_level_sizes, h_subtree_sizes);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
}

void terminate() {
    free(h_level_sizes);
    free(h_subtree_sizes);
    cudaFree(d_level_sizes);
    cudaFree(d_subtree_sizes);
    cudaFree(last);
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
                    const pos64 black_kings) {
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

void end_searching(pos64 * white_pawns,
                    pos64 * white_bishops,
                    pos64 * white_knights,
                    pos64 * white_rooks,
                    pos64 * white_queens,
                    pos64 * white_kings,
                    pos64 * black_pawns,
                    pos64 * black_bishops,
                    pos64 * black_knights,
                    pos64 * black_rooks,
                    pos64 * black_queens,
                    pos64 * black_kings,
                    pos64 * white_pawns_boards,
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
                    int * d_last) {
    int last;
    cudaMemcpy(&last, d_last, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(white_pawns, white_pawns_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(white_bishops, white_bishops_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(white_knights, white_knights_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(white_rooks, white_rooks_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(white_queens, white_queens_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(white_kings, white_kings_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_pawns, black_pawns_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_bishops, black_bishops_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_knights, black_knights_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_rooks, black_rooks_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_queens, black_queens_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(black_kings, black_kings_boards + last + 1, sizeof(pos64), cudaMemcpyDeviceToHost);
}


__global__ void run_first_stage(pos64 * white_pawns_boards,
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
                int level,
                short current_player,
                int * level_sizes,
                int * subtree_sizes,
                int basic_offset = 0) {
    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= level_sizes[level]) return;
    int index_offset = (level == 0 ? 0 : subtree_sizes[level - 1]) + basic_offset;
    int kids_offset = subtree_sizes[level] + index * BOARDS_GENERATED + basic_offset;
    current_player ^= (level & 1);

    DBG(if (index != 0) {return;})
    DBG(printf("Generuje ruchy gracza %d od pozycji %d i current_player %d\n", index + index_offset, kids_offset, current_player));

    generate_moves(&white_pawns_boards[index + index_offset],
                    &white_bishops_boards[index + index_offset],
                    &white_knights_boards[index + index_offset],
                    &white_rooks_boards[index + index_offset],
                    &white_queens_boards[index + index_offset],
                    &white_kings_boards[index + index_offset],
                    &black_pawns_boards[index + index_offset],
                    &black_bishops_boards[index + index_offset],
                    &black_knights_boards[index + index_offset],
                    &black_rooks_boards[index + index_offset],
                    &black_queens_boards[index + index_offset],
                    &black_kings_boards[index + index_offset],
                    &white_pawns_boards[kids_offset],
                    &white_bishops_boards[kids_offset],
                    &white_knights_boards[kids_offset],
                    &white_rooks_boards[kids_offset],
                    &white_queens_boards[kids_offset],
                    &white_kings_boards[kids_offset],
                    &black_pawns_boards[kids_offset],
                    &black_bishops_boards[kids_offset],
                    &black_knights_boards[kids_offset],
                    &black_rooks_boards[kids_offset],
                    &black_queens_boards[kids_offset],
                    &black_kings_boards[kids_offset],
                    current_player);
}

__global__ void run_first_stage_results(int * results,
                int level,
                short current_player,
                int * level_sizes,
                int * subtree_sizes,
                int * last,
                int basic_offset = 0) {
    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= level_sizes[level]) return;
    int index_offset = (level == 0 ? 0 : subtree_sizes[level - 1]) + basic_offset;
    int kids_offset = subtree_sizes[level] + index * BOARDS_GENERATED + basic_offset;
    current_player ^= (level & 1);

    DBG(if (index != 0) {return;})
    // DBG(printf("Zbieram wyniki gracza %d od pozycji %d i current_player %d i czy maksymalizuje? %d\n", 
    //     index + index_offset, kids_offset, current_player, current_player == WHITE));

    gather_results(&results[index + index_offset], &results[kids_offset], current_player == WHITE, last);
}

__global__ void run_first_stage_evaluate(pos64 * white_pawns_boards,
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
                int * level_sizes,
                int * subtree_sizes,
                int * results,
                int basic_offset = 0) {
    int level = MAX_DEPTH - FIRST_STAGE_DEPTH;
    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= level_sizes[level]) return;
    int index_offset = (level == 0 ? 0 : subtree_sizes[level - 1]) + basic_offset;
    
    DBG(if (index != 0) {return;})

    if ((white_kings_boards[index + index_offset] | black_kings_boards[index + index_offset]) == 0) {
        results[index + index_offset] = INF;    
    }
    else {
        results[index + index_offset] = evaluate_position(white_pawns_boards[index + index_offset],
                                                        white_bishops_boards[index + index_offset],
                                                        white_knights_boards[index + index_offset],
                                                        white_rooks_boards[index + index_offset],
                                                        white_queens_boards[index + index_offset],
                                                        white_kings_boards[index + index_offset],
                                                        black_pawns_boards[index + index_offset], 
                                                        black_bishops_boards[index + index_offset], 
                                                        black_knights_boards[index + index_offset],
                                                        black_rooks_boards[index + index_offset],
                                                        black_queens_boards[index + index_offset],
                                                        black_kings_boards[index + index_offset]);
    }
    DBG(printf("Evaluuje [%d] %d i mam %d\n", index, index + index_offset, results[index + index_offset]));
}

__global__ void copy_result(int * results, int from, int to) {
    DBG(printf("Kopiuje wynik z %d do %d\n", from, to));
    results[to] = results[from];
}

__global__ void copy_data(pos64 * white_pawns_boards,
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
                int from, int to) {
    DBG(printf("Kopiuje dane z %d do %d\n", from, to));
    white_pawns_boards[to] = white_pawns_boards[from];
    white_bishops_boards[to] = white_bishops_boards[from];
    white_knights_boards[to] = white_knights_boards[from];
    white_rooks_boards[to] = white_rooks_boards[from];
    white_queens_boards[to] = white_queens_boards[from];
    white_kings_boards[to] = white_kings_boards[from];
    black_pawns_boards[to] = black_pawns_boards[from];
    black_bishops_boards[to] = black_bishops_boards[from];
    black_knights_boards[to] = black_knights_boards[from];
    black_rooks_boards[to] = black_rooks_boards[from];
    black_queens_boards[to] = black_queens_boards[from];
    black_kings_boards[to] = black_kings_boards[from];
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
    int * results;
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
                black_kings);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    // generating moves in first stage
    DBG(printf("Stage 1 - generating moves\n"));
    for (int i = 0; i < FIRST_STAGE_DEPTH; i++) {
        run_first_stage<<<BLOCKS, THREADS>>>(white_pawns_boards,
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
                    i,
                    current_player,
                    d_level_sizes,
                    d_subtree_sizes);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }
    DBG(printf("Stage finished successfully\n"));

    // second stage
    int basic_offset = h_subtree_sizes[FIRST_STAGE_DEPTH] + 1;
    int player_offset = h_subtree_sizes[FIRST_STAGE_DEPTH - 1];
    for (int o = 0; o < h_level_sizes[FIRST_STAGE_DEPTH - 1]; o++) {

        copy_data<<<1, 1>>>(white_pawns_boards,
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
                    player_offset + o,
                    basic_offset);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        DBG(printf("Stage 2 - generating moves\n"));
        //generating moves
        for (int i = 0; i < MAX_DEPTH - FIRST_STAGE_DEPTH; i++) {
            run_first_stage<<<BLOCKS, THREADS>>>(white_pawns_boards,
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
                    i,
                    current_player ^ (i & 1) ^ ((FIRST_STAGE_DEPTH + i) & 1),
                    d_level_sizes,
                    d_subtree_sizes,
                    basic_offset);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());
        }
        DBG(printf("Stage finished successfully\n"));

        DBG(printf("Stage 2 - evaluating\n"));
        // evaluating
        run_first_stage_evaluate<<<BLOCKS, THREADS>>>(white_pawns_boards,
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
                    d_level_sizes,
                    d_subtree_sizes,
                    results,
                    basic_offset);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        DBG(printf("Stage finished successfully\n"));

        DBG(printf("Stage 2 - gathering results\n"));
        // gathering results
        for (int i = MAX_DEPTH - FIRST_STAGE_DEPTH - 1; i >= 0 ; i--) {
             run_first_stage_results<<<BLOCKS, THREADS>>>(results,
                    i,
                    current_player ^ (i & 1) ^ ((FIRST_STAGE_DEPTH + i) & 1),
                    d_level_sizes,
                    d_subtree_sizes,
                    last,
                    basic_offset);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());   
        }
        DBG(printf("Stage finished successfully\n"));

        copy_result<<<1, 1>>>(results, basic_offset, player_offset + o);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }

    DBG(printf("Stage 1 - gathering results\n"));
    // acquiring results for first stage
     for (int i = FIRST_STAGE_DEPTH; i >= 0; i--) {
        run_first_stage_results<<<BLOCKS, THREADS>>>(results,
                    i,
                    current_player,
                    d_level_sizes,
                    d_subtree_sizes,
                    last);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }
    DBG(printf("Stage finished successfully\n"));
    
    end_searching(
        &white_pawns,
        &white_bishops,
        &white_knights,
        &white_rooks,
        &white_queens,
        &white_kings,
        &black_pawns,
        &black_bishops,
        &black_knights,
        &black_rooks,
        &black_queens,
        &black_kings,
        white_pawns_boards,
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
        last);

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
}




















