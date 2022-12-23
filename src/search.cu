#include <algorithm>

#include "search.cuh"
#include "macros.cuh"
#include "evaluate.cuh"
#include "moves.cuh"
#include "scan.cuh"
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
int * last;

__device__ void gather_results(int* results_to, int* results_from, bool maximize, int * last, int boardCount) {
    int result;
    if (maximize) { // maximizing
        result = -INF;
        for (int i = 0; i < boardCount; i++) {
            if (results_from[i] != INF && results_from[i] != -INF && results_from[i] > result) {
                result = results_from[i];
                *last = i;
            }
        }
    } 
    else { // minimizing
        result = INF;
        for (int i = 0; i < boardCount; i++) {
           if (results_from[i] != INF && results_from[i] != -INF && results_from[i] < result) {
                result = results_from[i];
                *last = i;
            }
        }
    }
    *results_to = result;
}  

void init() {
    cudaSetDevice(0);
    CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));
}

void terminate() {
    cudaSetDevice(0);
    cudaFree(last);
}

dim3 getBlocksCount2d(int boards) {
    return dim3(MAX_BLOCKS_PER_DIMENSION, boards % MAX_BLOCKS_PER_DIMENSION == 0 ? boards / MAX_BLOCKS_PER_DIMENSION : boards / MAX_BLOCKS_PER_DIMENSION + 1);
}

void setThreadAndBlocksCount(int * threads, dim3 * blocks, int boardCount)
{
    if(boardCount <= MAX_THREADS) {
        *threads = boardCount;
        *blocks = dim3(1);
    }
    else if((boardCount / MAX_THREADS) + 1 < MAX_BLOCKS_PER_DIMENSION){
        *threads = MAX_THREADS;
        *blocks = dim3((boardCount / MAX_THREADS) + 1);
    }
    else
    {
        *threads = MAX_THREADS;
        *blocks = getBlocksCount2d(boardCount); 
    }
}

__global__ void generate_moves_for_boards(pos64 * boards, 
                unsigned int *boardsOffsets,
                bool isWhite,
                int boards_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boards_count) return; 
    
    pos64 *parent_destination = boards + (index * BOARD_SIZE);
    pos64 *kids_destination = boards + (boards_count * BOARD_SIZE) + (boardsOffsets[index] * BOARD_SIZE);

    generate_moves(parent_destination, kids_destination, isWhite);
}

__global__ void gather_results_for_boards(int * results,
                unsigned int *boardsOffsets,
                unsigned int currentLevelBoardCount,
                unsigned int lowerLevelBoardCount, // count of lower level 
                bool maximizing,
                int *last) {
    pos64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= currentLevelBoardCount) return;
    int *parent_destination = results + index;
    int *kids_destination = results + currentLevelBoardCount + boardsOffsets[index];

    unsigned int kidsBoardCount;
    if(index == (currentLevelBoardCount - 1))
    {
        kidsBoardCount = lowerLevelBoardCount - boardsOffsets[index];
    }
    else
    {
        kidsBoardCount = boardsOffsets[index + 1] - boardsOffsets[index];
    }

    gather_results(parent_destination, kids_destination, maximizing, last,  kidsBoardCount);
}

__global__ void evaluate_boards(pos64 * boards,
                unsigned int boardCount,
                int * results) {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int index = blockId * blockDim.x + threadIdx.x;
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

__global__ void pre_calculate_boards_count(pos64 *boards, unsigned int *boardsOffsets, bool isWhite, int boardCount) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= boardCount) return;

    boardsOffsets[index] = pre_count_moves(boards + (index * BOARD_SIZE), isWhite);
}

int prepareMemory(pos64 **boards, unsigned int **offsets, unsigned int **level_sizes) {
    gpuErrchk(cudaMallocManaged(level_sizes, sizeof(int) * MAX_POSSIBLE_DEPTH));

    size_t sizeOfOneBoard = sizeof(pos64) * BOARD_SIZE + sizeof(unsigned int);
    size_t total,free;
    cudaMemGetInfo(&free, &total);

    int maxBoardCount = free / (sizeOfOneBoard + 1);
    DBG(printf("max boards: %d\n", maxBoardCount));

    gpuErrchk(cudaMalloc(boards,  sizeof(pos64) * BOARD_SIZE * maxBoardCount));
    gpuErrchk(cudaMalloc(offsets, sizeof(unsigned int) * maxBoardCount));
    gpuErrchk(cudaMemset(*offsets, 0, sizeof(int) * maxBoardCount));
    gpuErrchk(cudaMemset(*boards, 0, sizeof(pos64) * BOARD_SIZE * maxBoardCount));

    *level_sizes[0] = 1;

    return maxBoardCount;
}

int runBoardGeneration(pos64 *boards, unsigned int *boardsOffsets, unsigned int *level_sizes, int *depthFound, bool *isWhite, int maxBoardsCount, bool isFirstStage) {
    int runningBoards, boardOffset = 0;
    int threadCount;
    dim3 blockCount;
    int offset = 0;
    *depthFound = MAX_POSSIBLE_DEPTH;
    for(int i = 0; i < MAX_POSSIBLE_DEPTH; i++) 
    {
        runningBoards = level_sizes[i];

        DBG(printf("generating depth: %d, evaluated boards: %d\n", i + 1, runningBoards));

        // first stage - check how many boards will be generated for each board
        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        pre_calculate_boards_count<<<blockCount, threadCount>>>(boards + offset * BOARD_SIZE, boardsOffsets + offset, isWhite, runningBoards);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        // secound stage - find boardsOffsets for each board to put their kids there
        DBG(printf("calculate offsets offset: %d\n", offset));
        scan(boardsOffsets + offset, runningBoards, (unsigned int*)(boards + (offset + runningBoards) * BOARD_SIZE), &level_sizes[i + 1]); // since boards are not yet created I use the space there as a temp table

        DBG(printf("boardCount on depth %d, %u\n", i, level_sizes[i + 1]));

        if((isFirstStage && level_sizes[i + 1] > MAX_BOARD_COMPUTED_IN_SECOUND_STAGE) || (!isFirstStage && runningBoards + offset + level_sizes[i + 1] > maxBoardsCount)) {
            *depthFound = i;
            break;
        }
        // third stage - generate boards

        DBG(printf("generate boards\n"));
        generate_moves_for_boards<<<blockCount, threadCount>>>(boards + offset * BOARD_SIZE, boardsOffsets + offset, *isWhite, runningBoards);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        *isWhite = !(*isWhite);
        offset += runningBoards;
    }
    return offset;
}

void gatherResults(pos64 *boards, unsigned int *boardsOffsets, unsigned int *level_sizes, int depthFound, int* localLast, int offsetToLastLevel, bool maximizing) {
    int threadCount, runningBoards;
    dim3 blockCount;
    int offset = offsetToLastLevel;

    for (int i = depthFound - 1; i >= 0; i--) {
        runningBoards = level_sizes[i];
        offset -= runningBoards;

        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        gather_results_for_boards<<<blockCount, threadCount>>>((int*)(boardsOffsets + offset), boardsOffsets + offset, runningBoards, level_sizes[i + 1], maximizing, localLast); // since each thread uses the offset only once and then writes to one place i can just swap the values here
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        maximizing = !maximizing;
    }
}


void search(const short& current_player,
            pos64 *position) {
    
    unsigned int *level_sizes, *boardsOffsets;
    pos64 *boards;
    size_t totalBoardsCount = prepareMemory(&boards, &boardsOffsets, &level_sizes);
    
    gpuErrchk(cudaMemcpy(boards, position, sizeof(pos64) * BOARD_SIZE, cudaMemcpyHostToDevice));

    int firstStageDepth;
    bool isWhite = current_player == WHITE;
    int offset = runBoardGeneration(boards, boardsOffsets, level_sizes, &firstStageDepth, &isWhite, totalBoardsCount, true);
    // second stage

    // copying data
    int totalBoardCountInFirstStage = offset + level_sizes[firstStageDepth];
    pos64 *firstStageBoards = new pos64[totalBoardCountInFirstStage * BOARD_SIZE];
    int *firstStageOffsets = new int[totalBoardCountInFirstStage];

    gpuErrchk(cudaMemcpy(firstStageBoards, boards, sizeof(pos64) * BOARD_SIZE * totalBoardCountInFirstStage, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(firstStageOffsets, boardsOffsets, sizeof(int) * totalBoardCountInFirstStage, cudaMemcpyDeviceToHost));
    
    cudaFree(boards);
    cudaFree(boardsOffsets);


    std::vector<std::thread> threads;
    int devices_count;
    cudaGetDeviceCount(&devices_count);

    int boardsToCalaculateInSecStage = level_sizes[firstStageDepth];

    DBG(printf("Stage two started\n"));
    for (int j = 0; j < devices_count; j++) {
        threads.push_back (std::thread ([&, j, boardsToCalaculateInSecStage , isWhite, devices_count] () {
            gpuErrchk(cudaSetDevice(j));

            pos64 *secStageBoards;
            unsigned int *secStageOffsets, *sec_stage_level_sizes;
            int * last;
            CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));
            
            int maxBoards = prepareMemory(&secStageBoards, &secStageOffsets, &sec_stage_level_sizes);
            pos64 *baseBoardsAddress = firstStageBoards + offset * BOARD_SIZE;
            int *baseOffsetsAddress = firstStageOffsets + offset;
            for (int o = j; o < boardsToCalaculateInSecStage; o += devices_count) {
                gpuErrchk(cudaMemset(secStageOffsets, 0, sizeof(int) * maxBoards));
                gpuErrchk(cudaMemset(secStageBoards, 0, sizeof(pos64) * BOARD_SIZE * maxBoards));
                gpuErrchk(cudaMemcpy(secStageBoards, baseBoardsAddress + (o * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyHostToDevice));
                
                int depthFound;
                bool isWhiteTemp = isWhite;
                int tempOffset = runBoardGeneration(secStageBoards, secStageOffsets, sec_stage_level_sizes, &depthFound, &isWhiteTemp, maxBoards, false);
                DBG(printf("offset %d, depthFound: %d\n", tempOffset, depthFound));
                // evaluating

                DBG(printf("count of evaluation boards: %u\n", sec_stage_level_sizes[depthFound]));
                evaluate_boards<<<getBlocksCount2d(sec_stage_level_sizes[depthFound]), MAX_THREADS>>>(boards + tempOffset * BOARD_SIZE, sec_stage_level_sizes[depthFound], (int*)(boardsOffsets + tempOffset)); // since last level doesnt use offsets board i use it for keeping evaluation            
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaPeekAtLastError());

                gatherResults(secStageBoards, secStageOffsets, sec_stage_level_sizes, depthFound, last, tempOffset, !isWhiteTemp);

                gpuErrchk(cudaMemcpy(baseOffsetsAddress +  o, secStageOffsets, sizeof(int), cudaMemcpyDeviceToHost));
            }
            cudaFree(secStageBoards);
            cudaFree(secStageOffsets);
            cudaFree(sec_stage_level_sizes);
            cudaFree(last);
        }));
    }
    for (int j = 0; j < devices_count; j++) {
        threads[j].join();
    }
    cudaSetDevice(0);

    DBG(printf("Stage 1 - gathering results\n"));
    CHECK_ALLOC(cudaMalloc(&boards, sizeof(pos64) * BOARD_SIZE * totalBoardCountInFirstStage));
    CHECK_ALLOC(cudaMalloc(&boardsOffsets, sizeof(int) * totalBoardCountInFirstStage));

    gpuErrchk(cudaMemcpy(boards, firstStageBoards, sizeof(pos64) * BOARD_SIZE * totalBoardCountInFirstStage, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(boardsOffsets, firstStageOffsets, sizeof(int) * totalBoardCountInFirstStage, cudaMemcpyHostToDevice));

    delete[] firstStageBoards;
    delete[] firstStageOffsets;

    // acquiring results for first stage
    gatherResults(boards, boardsOffsets, level_sizes, firstStageDepth, last, offset, !isWhite);

    int bestMoveNr;
    gpuErrchk(cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d\n", bestMoveNr);
    gpuErrchk(cudaMemcpy(position, boards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));
    cudaFree(level_sizes);

    cudaFree(boards);
    cudaFree(boardsOffsets);
}