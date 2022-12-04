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
int * h_subtree_sizes;
int * last;

__device__ void gather_results(int* results_to, int* results_from, bool maximize, int * last) {
    int result;
    int couter = 0;
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
    subtree_sizes[MAX_DEPTH + 1] = level_sizes[MAX_DEPTH]  + subtree_sizes[MAX_DEPTH];

}

void init() {
    h_level_sizes = new int[MAX_DEPTH];
    h_subtree_sizes = new int[MAX_DEPTH];
    CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));

    _init_sizes_tables(h_level_sizes, h_subtree_sizes);
}

void terminate() {
    free(h_level_sizes);
    free(h_subtree_sizes);
    cudaFree(last);
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
    if (board_address[WHITE_KING_OFFSET] == 0 | board_address[BLACK_KING_OFFSET] == 0) { // is it properly handled ??
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
    
    pos64 *firstStageBoards, *secStageBoards;
    int* firstStageResults, *secStageResult;
    int firstStageTotalBoardsCount  = h_subtree_sizes[FIRST_STAGE_DEPTH];
    int secStageTotalBoardsCount = h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH];
    CHECK_ALLOC(cudaMalloc(&firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount));
    CHECK_ALLOC(cudaMalloc(&firstStageResults, sizeof(int) * firstStageTotalBoardsCount));
    CHECK_ALLOC(cudaMalloc(&secStageBoards, sizeof(pos64) * BOARD_SIZE * secStageTotalBoardsCount));
    CHECK_ALLOC(cudaMalloc(&secStageResult, sizeof(int) * secStageTotalBoardsCount));

    gpuErrchk(cudaMemcpy(firstStageBoards, position, BOARD_SIZE * sizeof(pos64), cudaMemcpyHostToDevice));

    // generating moves in first stage
    DBG(printf("Stage 1 - generating moves\n"));
    
    pos64* firstBoardAddress;
    bool isWhite = current_player == WHITE;
    for (int i = 0; i < FIRST_STAGE_DEPTH; i++) {
        if(i == 0) {
            firstBoardAddress = firstStageBoards;
        }
        else {
            firstBoardAddress = firstStageBoards + (h_subtree_sizes[i - 1] * BOARD_SIZE);
        } 
        generate_moves_for_boards<<<BLOCKS, THREADS>>>(firstStageBoards, isWhite, h_level_sizes[i]);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        isWhite = !isWhite;
    }
    DBG(printf("Stage finished successfully\n"));

    // second stage
    pos64 * basicBoardAddres = firstStageBoards + (h_subtree_sizes[FIRST_STAGE_DEPTH - 1] * BOARD_SIZE); 
    int *basicResultAddress = firstStageResults + h_subtree_sizes[FIRST_STAGE_DEPTH - 1];
    for (int o = 0; o < h_level_sizes[FIRST_STAGE_DEPTH]; o++) {        
        gpuErrchk(cudaMemcpy(secStageBoards, basicBoardAddres + (o * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToDevice));

        DBG(printf("Stage 2 - generating moves\n"));
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
        DBG(printf("Stage finished successfully\n"));

        DBG(printf("Stage 2 - evaluating\n"));
        // evaluating

        pos64 *lowestLevelAdress = secStageBoards + (h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH - 1] * BOARD_SIZE);
        evaluate_boards<<<BLOCKS, THREADS>>>(lowestLevelAdress, h_level_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH], secStageResult + h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH - 1]);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        DBG(printf("Stage finished successfully\n"));

        DBG(printf("Stage 2 - gathering results\n"));
        
        for (int i = MAX_DEPTH - FIRST_STAGE_DEPTH - 1; i >= 0 ; i--) {
            gather_results_for_boards<<<BLOCKS, THREADS>>>(secStageResult, h_level_sizes[i], !isWhiteTemp , last);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());   
            isWhiteTemp = !isWhiteTemp;
        }
        DBG(printf("Stage finished successfully\n"));

        gpuErrchk(cudaMemcpy(basicResultAddress +  o, secStageResult, sizeof(int), cudaMemcpyDeviceToDevice));
    }

    DBG(printf("Stage 1 - gathering results\n"));
    // acquiring results for first stage
     for (int i = FIRST_STAGE_DEPTH - 1; i >= 0; i--) {
        gather_results_for_boards<<<BLOCKS, THREADS>>>(firstStageResults, h_level_sizes[i], !isWhite, last);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        isWhite = !isWhite;
    }
    DBG(printf("Stage finished successfully\n"));

    int bestMoveNr;
    gpuErrchk(cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(position, firstStageBoards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(firstStageBoards);
    cudaFree(secStageBoards);
    cudaFree(firstStageResults);
    cudaFree(secStageResult);
}