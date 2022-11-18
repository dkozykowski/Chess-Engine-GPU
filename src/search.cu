#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>

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
int * d_level_sizes[MAX_CARDS];
int * h_subtree_sizes;
int * d_subtree_sizes[MAX_CARDS];
int * last[MAX_CARDS];

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
    int devices_count;
    cudaGetDeviceCount(&devices_count);
    for (int i = 0; i < devices_count; i++) {
        cudaSetDevice(i);
        CHECK_ALLOC(cudaMalloc(&d_level_sizes[i], sizeof(int) * MAX_DEPTH + 1));
        CHECK_ALLOC(cudaMalloc(&d_subtree_sizes[i], sizeof(int) * MAX_DEPTH + 2));
        CHECK_ALLOC(cudaMalloc(&last[i], sizeof(int)));
        init_sizes_tables<<<1, 1>>>(d_level_sizes[i], d_subtree_sizes[i]);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }

    _init_sizes_tables(h_level_sizes, h_subtree_sizes);
}

void terminate() {
    free(h_level_sizes);
    free(h_subtree_sizes);
    int devices_count;
    cudaGetDeviceCount(&devices_count);
    for (int i = 0; i < devices_count; i++) {
        cudaSetDevice(i);
        cudaFree(d_level_sizes[i]);
        cudaFree(d_subtree_sizes[i]);
        cudaFree(last[i]);
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
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

void unset_device_and_copy_result(int * results_0, int * results, int from, int to) {
    int result;
    cudaMemcpy(&result, &results[from], sizeof(int), cudaMemcpyDeviceToHost);
    cudaSetDevice(0);
    cudaMemcpy(&results_0[to], &result, sizeof(int), cudaMemcpyHostToDevice);
}

void set_device_and_copy_data(pos64 * white_pawns_boards_0,
                pos64 * white_bishops_boards_0,
                pos64 * white_knights_boards_0,
                pos64 * white_rooks_boards_0,
                pos64 * white_queens_boards_0,
                pos64 * white_kings_boards_0,
                pos64 * black_pawns_boards_0,
                pos64 * black_bishops_boards_0,
                pos64 * black_knights_boards_0,
                pos64 * black_rooks_boards_0,
                pos64 * black_queens_boards_0,
                pos64 * black_kings_boards_0,
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
                int from, int to,
                int device_id) {
    int temp[16];
    cudaMemcpy(&temp[0], &white_pawns_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[1], &white_bishops_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[2], &white_knights_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[3], &white_rooks_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[4], &white_queens_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[5], &white_kings_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[6], &black_pawns_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[7], &black_bishops_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[8], &black_knights_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[9], &black_rooks_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[10], &black_queens_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp[11], &black_kings_boards_0[from], sizeof(pos64), cudaMemcpyDeviceToHost);

    cudaSetDevice(device_id);

    cudaMemcpy(&white_pawns_boards[from], &temp[0], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&white_bishops_boards[from], &temp[1], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&white_knights_boards[from], &temp[2], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&white_rooks_boards[from], &temp[3], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&white_queens_boards[from], &temp[4], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&white_kings_boards[from], &temp[5], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&black_pawns_boards[from], &temp[6], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&black_bishops_boards[from], &temp[7], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&black_knights_boards[from], &temp[8], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&black_rooks_boards[from], &temp[9], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&black_queens_boards[from], &temp[10], sizeof(pos64), cudaMemcpyHostToDevice);
    cudaMemcpy(&black_kings_boards[from], &temp[11], sizeof(pos64), cudaMemcpyHostToDevice);
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
        
    pos64*  white_pawns_boards[MAX_CARDS];
    pos64*  white_bishops_boards[MAX_CARDS];
    pos64*  white_knights_boards[MAX_CARDS];
    pos64*  white_rooks_boards[MAX_CARDS];
    pos64*  white_queens_boards[MAX_CARDS];
    pos64*  white_kings_boards[MAX_CARDS];
    pos64*  black_pawns_boards[MAX_CARDS];
    pos64*  black_bishops_boards[MAX_CARDS];
    pos64*  black_knights_boards[MAX_CARDS];
    pos64*  black_rooks_boards[MAX_CARDS];
    pos64*  black_queens_boards[MAX_CARDS];
    pos64*  black_kings_boards[MAX_CARDS];
    int * results[MAX_CARDS];

    std::vector<std::thread> threads;
    int devices_count;
    cudaGetDeviceCount(&devices_count);
    for (int i = 0; i < devices_count; i++) {
        cudaSetDevice(i);
        CHECK_ALLOC(cudaMalloc(&white_pawns_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&white_bishops_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&white_knights_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&white_rooks_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&white_queens_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&white_kings_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&black_pawns_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&black_bishops_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&black_knights_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&black_rooks_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&black_queens_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&black_kings_boards[i], sizeof(pos64) * MAX_BOARDS_IN_MEMORY));
        CHECK_ALLOC(cudaMalloc(&results[i], sizeof(int) * MAX_BOARDS_IN_MEMORY));
    }

    cudaSetDevice(0);
    init_searching<<<1, 1>>>(white_pawns_boards[0],
                white_bishops_boards[0],
                white_knights_boards[0],
                white_rooks_boards[0],
                white_queens_boards[0],
                white_kings_boards[0],
                black_pawns_boards[0],
                black_bishops_boards[0],
                black_knights_boards[0],
                black_rooks_boards[0],
                black_queens_boards[0],
                black_kings_boards[0],
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
        run_first_stage<<<BLOCKS, THREADS>>>(white_pawns_boards[0],
                    white_bishops_boards[0],
                    white_knights_boards[0],
                    white_rooks_boards[0],
                    white_queens_boards[0],
                    white_kings_boards[0],
                    black_pawns_boards[0],
                    black_bishops_boards[0],
                    black_knights_boards[0],
                    black_rooks_boards[0],
                    black_queens_boards[0],
                    black_kings_boards[0],
                    i,
                    current_player,
                    d_level_sizes[0],
                    d_subtree_sizes[0]);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }
    DBG(printf("Stage finished successfully\n"));

    // second stage
    int basic_offset = h_subtree_sizes[FIRST_STAGE_DEPTH] + 1;
    int player_offset = h_subtree_sizes[FIRST_STAGE_DEPTH - 1];
    for (int j = 0; j < devices_count; j++) {
        threads.push_back (std::thread ([&,j] () {
            for (int o = j; o < h_level_sizes[FIRST_STAGE_DEPTH - 1]; o += devices_count) 
            {
                set_device_and_copy_data(white_pawns_boards[0],
                        white_bishops_boards[0],
                        white_knights_boards[0],
                        white_rooks_boards[0],
                        white_queens_boards[0],
                        white_kings_boards[0],
                        black_pawns_boards[0],
                        black_bishops_boards[0],
                        black_knights_boards[0],
                        black_rooks_boards[0],
                        black_queens_boards[0],
                        black_kings_boards[0],
                        white_pawns_boards[j],
                        white_bishops_boards[j],
                        white_knights_boards[j],
                        white_rooks_boards[j],
                        white_queens_boards[j],
                        white_kings_boards[j],
                        black_pawns_boards[j],
                        black_bishops_boards[j],
                        black_knights_boards[j],
                        black_rooks_boards[j],
                        black_queens_boards[j],
                        black_kings_boards[j], 
                        player_offset + o,
                        basic_offset,
                        j);

                //generating moves
                for (int i = 0; i < MAX_DEPTH - FIRST_STAGE_DEPTH; i++) {
                    run_first_stage<<<BLOCKS, THREADS>>>(white_pawns_boards[j],
                            white_bishops_boards[j],
                            white_knights_boards[j],
                            white_rooks_boards[j],
                            white_queens_boards[j],
                            white_kings_boards[j],
                            black_pawns_boards[j],
                            black_bishops_boards[j],
                            black_knights_boards[j],
                            black_rooks_boards[j],
                            black_queens_boards[j],
                            black_kings_boards[j],
                            i,
                            current_player ^ (i & 1) ^ ((FIRST_STAGE_DEPTH + i) & 1),
                            d_level_sizes[j],
                            d_subtree_sizes[j],
                            basic_offset);
                    gpuErrchk(cudaDeviceSynchronize());
                    gpuErrchk(cudaPeekAtLastError());
                }

                // evaluating
                run_first_stage_evaluate<<<BLOCKS, THREADS>>>(white_pawns_boards[j],
                            white_bishops_boards[j],
                            white_knights_boards[j],
                            white_rooks_boards[j],
                            white_queens_boards[j],
                            white_kings_boards[j],
                            black_pawns_boards[j],
                            black_bishops_boards[j],
                            black_knights_boards[j],
                            black_rooks_boards[j],
                            black_queens_boards[j],
                            black_kings_boards[j],
                            d_level_sizes[j],
                            d_subtree_sizes[j],
                            results[j],
                            basic_offset);
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaPeekAtLastError());
                
                // gathering results
                for (int i = MAX_DEPTH - FIRST_STAGE_DEPTH - 1; i >= 0 ; i--) {
                    run_first_stage_results<<<BLOCKS, THREADS>>>(results[j],
                            i,
                            current_player ^ (i & 1) ^ ((FIRST_STAGE_DEPTH + i) & 1),
                            d_level_sizes[j],
                            d_subtree_sizes[j],
                            last[j],
                            basic_offset);
                    gpuErrchk(cudaDeviceSynchronize());
                    gpuErrchk(cudaPeekAtLastError());   
                }
                unset_device_and_copy_result(results[0], results[j], basic_offset, player_offset + o);
            }
        }));
    }

    DBG(printf("Stage 1 - gathering results\n"));
    // acquiring results for first stage
     for (int i = FIRST_STAGE_DEPTH; i >= 0; i--) {
        run_first_stage_results<<<BLOCKS, THREADS>>>(results[0],
                    i,
                    current_player,
                    d_level_sizes[0],
                    d_subtree_sizes[0],
                    last[0]);
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
        white_pawns_boards[0],
        white_bishops_boards[0],
        white_knights_boards[0],
        white_rooks_boards[0],
        white_queens_boards[0],
        white_kings_boards[0],
        black_pawns_boards[0],
        black_bishops_boards[0],
        black_knights_boards[0],
        black_rooks_boards[0],
        black_queens_boards[0],
        black_kings_boards[0],
        last[0]);

    for (int i = 0; i < devices_count; i++) {
        cudaSetDevice(i);
        cudaFree(white_pawns_boards[i]);
        cudaFree(white_bishops_boards[0]);
        cudaFree(white_knights_boards[0]);
        cudaFree(white_rooks_boards[0]);
        cudaFree(white_queens_boards[0]);
        cudaFree(white_kings_boards[0]);
        cudaFree(black_pawns_boards[0]);
        cudaFree(black_bishops_boards[0]);
        cudaFree(black_knights_boards[0]);
        cudaFree(black_rooks_boards[0]);
        cudaFree(black_queens_boards[0]);
        cudaFree(black_kings_boards[0]);
        cudaFree(results[0]);
    }
}
