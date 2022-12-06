#include <algorithm>

#include "search.cuh"
#include "macros.cuh"
#include "evaluate.cuh"
#include "moves.cuh"
#include "vector"
#include "thread"

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
int * h_subtree_sizes;
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

void _init_sizes_tables(int* level_sizes, int * subtree_sizes) {
    level_sizes[0] = 1;
    subtree_sizes[0] = 1;
    for (int i = 1;  i < MAX_DEPTH; i++) {
        level_sizes[i] = level_sizes[i - 1] * BOARDS_GENERATED;
        subtree_sizes[i] = level_sizes[i] + subtree_sizes[i - 1];
    }
}

void init() {
    cudaSetDevice(0);
    h_level_sizes = new int[MAX_DEPTH + 10];
    h_subtree_sizes = new int[MAX_DEPTH + 10];
    CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));

    _init_sizes_tables(h_level_sizes, h_subtree_sizes);
}

void terminate() {
    delete[] h_level_sizes;
    delete[] h_subtree_sizes;
    cudaSetDevice(0);
    cudaFree(last);
}

void copy_from_gpu_to_cpu(void * device_source, void * host_destination, int size) {
    gpuErrchk(cudaMemcpy(host_destination, device_source, size, cudaMemcpyDeviceToHost));
}

void copy_from_cpu_to_gpu(void * host_source, void * device_destination, int size) {
    gpuErrchk(cudaMemcpy(device_destination, host_source, size, cudaMemcpyHostToDevice));
}

__global__ void generate_moves_for_boards(pos64 * boards,
                bool isWhite,
                int boards_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boards_count) return; 
    
    pos64 *parent_destination = boards + (index * BOARD_SIZE);
    pos64 *kids_destination = boards + (boards_count * BOARD_SIZE) + (index * BOARDS_GENERATED * BOARD_SIZE);

    generate_moves(parent_destination, kids_destination, isWhite);
}

__global__ void gather_results_for_boards(int * results,
                int boardCount, 
                bool maximizing,
                int *last) {
    pos64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boardCount) return;
    int *parent_destination = results + index;
    int *kids_destination = results + boardCount + (index * BOARDS_GENERATED);

    gather_results(parent_destination, kids_destination, maximizing, last);
}

__global__ void evaluate_boards(pos64 * boards,
                int boardCount,
                int * results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boardCount) return;
    pos64 *board_address = boards + (index * BOARD_SIZE);
    if (board_address[WHITE_KING_OFFSET] == 0 && board_address[BLACK_KING_OFFSET] == 0) { // is it properly handled ??
        results[index] = INF;    
    }
    else {
        results[index] = evaluate_position(board_address[WHITE_PAWN_OFFSET],
                                        board_address[WHITE_BISHOP_OFFSET],
                                        board_address[WHITE_KNIGHT_OFFSET],
                                        board_address[WHITE_ROOK_OFFSET],
                                        board_address[WHITE_QUEEN_OFFSET],
                                        board_address[WHITE_KING_OFFSET],
                                        board_address[BLACK_PAWN_OFFSET],
                                        board_address[BLACK_BISHOP_OFFSET],
                                        board_address[BLACK_KNIGHT_OFFSET],
                                        board_address[BLACK_ROOK_OFFSET],
                                        board_address[BLACK_QUEEN_OFFSET],
                                        board_address[BLACK_KING_OFFSET]);
    }
}

void search(const short& current_player,
            const int& move_num,
            pos64 *position) {
    
    pos64 *firstStageBoards;
    int* firstStageResults;
    int firstStageTotalBoardsCount  = h_subtree_sizes[FIRST_STAGE_DEPTH];
    int secStageTotalBoardsCount = h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH];
    CHECK_ALLOC(cudaMalloc(&firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount));
    CHECK_ALLOC(cudaMalloc(&firstStageResults, sizeof(int) * firstStageTotalBoardsCount));

    pos64 *temp_firstStageBoards;
    int* temp_firstStageResults;
    try {
        temp_firstStageBoards = new pos64[BOARD_SIZE * firstStageTotalBoardsCount];
        temp_firstStageResults = new int[firstStageTotalBoardsCount];
    } catch (std::bad_alloc&) {
        ERR("Operator new");
    }

    gpuErrchk(cudaMemcpy(firstStageBoards, position, BOARD_SIZE * sizeof(pos64), cudaMemcpyHostToDevice));

    // generating moves in first stage
    DBG(printf("Stage 1 - generating moves\n"));
    
    bool isWhite = current_player == WHITE;
    for (int i = 0; i < FIRST_STAGE_DEPTH; i++) {
        generate_moves_for_boards<<<BLOCKS, THREADS>>>(firstStageBoards, isWhite, h_level_sizes[i]);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        isWhite = !isWhite;
    }
    DBG(printf("Stage finished successfully\n"));

    // second stage

    // copying data
    copy_from_gpu_to_cpu(firstStageBoards, temp_firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount);
    copy_from_gpu_to_cpu(firstStageResults, temp_firstStageResults, sizeof(int) * firstStageTotalBoardsCount);

    cudaFree(firstStageBoards);
    cudaFree(firstStageResults);

    std::vector<std::thread> threads;
    int devices_count;
    cudaGetDeviceCount(&devices_count);

    DBG(printf("Stage two started\n"));
    for (int j = 0; j < devices_count; j++) {
        threads.push_back (std::thread ([&, j, firstStageTotalBoardsCount, secStageTotalBoardsCount, isWhite, devices_count] () {
            gpuErrchk(cudaSetDevice(j));

            pos64* firstBoardAddress;
            pos64 *firstStageBoards, *secStageBoards;
            int* firstStageResults, *secStageResult;
            int * last;
            CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));
            CHECK_ALLOC(cudaMalloc(&firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount));
            CHECK_ALLOC(cudaMalloc(&firstStageResults, sizeof(int) * firstStageTotalBoardsCount));
            CHECK_ALLOC(cudaMalloc(&secStageBoards, sizeof(pos64) * BOARD_SIZE * secStageTotalBoardsCount));
            CHECK_ALLOC(cudaMalloc(&secStageResult, sizeof(int) * secStageTotalBoardsCount));

            pos64 * basicBoardAddres = temp_firstStageBoards + (h_subtree_sizes[FIRST_STAGE_DEPTH - 1] * BOARD_SIZE); 
            int *basicResultAddress = temp_firstStageResults + h_subtree_sizes[FIRST_STAGE_DEPTH - 1];

            for (int o = j; o < h_level_sizes[FIRST_STAGE_DEPTH]; o += devices_count) {
                copy_from_cpu_to_gpu(basicBoardAddres + (o * BOARD_SIZE), secStageBoards, sizeof(pos64) * BOARD_SIZE);
                
                //generating moves
                bool isWhiteTemp = isWhite;
                for (int i = 0; i < MAX_DEPTH - FIRST_STAGE_DEPTH; i++) {
                    if(i == 0) {
                        firstBoardAddress = secStageBoards;
                    }
                    else
                    {
                        firstBoardAddress = secStageBoards + (h_subtree_sizes[i - 1] * BOARD_SIZE);
                    }
                    generate_moves_for_boards<<<BLOCKS, THREADS>>>(firstBoardAddress, isWhiteTemp, h_level_sizes[i]);
                    gpuErrchk(cudaDeviceSynchronize());
                    gpuErrchk(cudaPeekAtLastError());
                    isWhiteTemp = !isWhiteTemp;
                }

                // evaluating
                pos64 *lowestLevelAdress = secStageBoards + (h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH - 1] * BOARD_SIZE);
                evaluate_boards<<<BLOCKS, THREADS>>>(lowestLevelAdress, h_level_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH], secStageResult + h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH - 1]);
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaPeekAtLastError());
                
                int *firstResultAddress;
                for (int i = MAX_DEPTH - FIRST_STAGE_DEPTH - 1; i >= 0 ; i--) {
                    if(i == 0) {
                        firstResultAddress = secStageResult;
                    }
                    else {
                        firstResultAddress = secStageResult + h_level_sizes[i - 1];
                    }
                    gather_results_for_boards<<<BLOCKS, THREADS>>>(firstResultAddress, h_level_sizes[i], !isWhiteTemp, last);
                    gpuErrchk(cudaDeviceSynchronize());
                    gpuErrchk(cudaPeekAtLastError());   
                    isWhiteTemp = !isWhiteTemp;
                }

                copy_from_gpu_to_cpu(secStageResult, basicResultAddress +  o, sizeof(int));
            }
            cudaFree(firstStageBoards);
            cudaFree(firstStageResults);
            cudaFree(secStageBoards);
            cudaFree(secStageResult);
            cudaFree(last);
        }));
    }
    for (int j = 0; j < devices_count; j++) {
        threads[j].join();
    }
    cudaSetDevice(0);

    DBG(printf("Stage 1 - gathering results\n"));
    CHECK_ALLOC(cudaMalloc(&firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount));
    CHECK_ALLOC(cudaMalloc(&firstStageResults, sizeof(int) * firstStageTotalBoardsCount));

    copy_from_cpu_to_gpu(temp_firstStageBoards, firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount);
    copy_from_cpu_to_gpu(temp_firstStageResults, firstStageResults, sizeof(int) * firstStageTotalBoardsCount);

    delete[] temp_firstStageBoards;
    delete[] temp_firstStageResults;

    // acquiring results for first stage
    int *firstResultAddress;
     for (int i = FIRST_STAGE_DEPTH - 1; i >= 0; i--) {
        if(i == 0) {
                firstResultAddress = firstStageResults;
            }
            else {
                firstResultAddress = firstStageResults + h_subtree_sizes[i - 1];
            }
        gather_results_for_boards<<<BLOCKS, THREADS>>>(firstResultAddress, h_subtree_sizes[i], !isWhite, last);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        isWhite = !isWhite;
    }
    DBG(printf("Stage finished successfully\n"));

    int bestMoveNr;
    gpuErrchk(cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(position, firstStageBoards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(firstStageBoards);
    cudaFree(firstStageResults);
}