#include <algorithm>
#include <iostream>


#include "evaluate.cuh"
#include "macros.cuh"
#include "moves.cuh"
#include "scan.cuh"
#include "search.cuh"
#include "thread"
#include "vector"

namespace SEARCH {

int *last;

/**
 * Calculates best moves for node based on precalculated results for sons.
 *
 * @param[out] resultsTo Pointer which found best result evaluation is written
 * to.
 * @param resultsFrom Pointer to array with precalculated evaluation results for
 * sons.
 * @param maximize Wheather the node shoud maximize or minimize sons
 * evaluations.
 * @param[out] last Pointer to global last value holding index of best son for
 * last node.
 * @param boardCount Number of sons.
 */
__device__ void gatherResults(int *resultsTo, int *resultsFrom, bool maximize,
                              int *last, int boardCount) {
    int result;
    if (maximize) {
        result = -INF;
        for (int i = 0; i < boardCount; i++) {
            if (resultsFrom[i] != INF && resultsFrom[i] != -INF &&
                resultsFrom[i] > result) {
                result = resultsFrom[i];
                *last = i;
            }
        }
    } else {
        result = INF;
        for (int i = 0; i < boardCount; i++) {
            if (resultsFrom[i] != INF && resultsFrom[i] != -INF &&
                resultsFrom[i] < result) {
                result = resultsFrom[i];
                *last = i;
            }
        }
    }
    *resultsTo = result;
}

/**
 * Allocates and initializes tables for search engine.
 */
void init() {
    cudaSetDevice(0);
    CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));
}

/**
 * Frees search engine tables.
 */
void terminate() {
    cudaSetDevice(0);
    cudaFree(last);
}

/**
 * Calculates optimal blocks and threads structure for given boards number.
 *
 * @param boards Number of boards.
 * @return dim3 structure holding blocks and threads number
 */
dim3 getBlocksCount2d(int boards) {
    return dim3(MAX_BLOCKS_PER_DIMENSION,
                boards % MAX_BLOCKS_PER_DIMENSION == 0
                    ? boards / MAX_BLOCKS_PER_DIMENSION
                    : boards / MAX_BLOCKS_PER_DIMENSION + 1);
}

/**
 * Calculates best moves for node based on precalculated results for sons.
 *
 * @param[out] threads Pointer to int storing optimal number of threads for
 * boardCount.
 * @param[out] blocks Pointer to int storing optimal number of blocks for
 * boardCount.
 * @param boardCount Number of boards.
 */
void setThreadAndBlocksCount(int *threads, dim3 *blocks, int boardCount) {
    if (boardCount <= MAX_THREADS) {
        *threads = boardCount;
        *blocks = dim3(1);
    } else if ((boardCount / MAX_THREADS) + 1 < MAX_BLOCKS_PER_DIMENSION) {
        *threads = MAX_THREADS;
        *blocks = dim3((boardCount / MAX_THREADS) + 1);
    } else {
        *threads = MAX_THREADS;
        *blocks = getBlocksCount2d(boardCount);
    }
}

/**
 * Generates all valid moves for given position.
 *
 * @param boards Pointer to array where board positions are stored. Points to
 * the first board of the boards, that is used to generate another level of the
 * three and childeren are stored behind parents in the array.
 * @param boardsOffset Pointer to array storing offsets where boards for each
 * node in @ref boards array starts.
 * @param isWhite Wheather the considered node is on odd level. (eg. true for
 * level 1, false for level 2, and so on).
 * @param boardsCount Number of sons that will be generated for considered node.
 */
__global__ void generateMovesForBoards(pos64 *boards,
                                       unsigned int *boardsOffsets,
                                       bool isWhite, int boardsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boardsCount) return;

    pos64 *parentDestination = boards + (index * BOARD_SIZE);
    pos64 *kidsDestination = boards + (boardsCount * BOARD_SIZE) +
                             (boardsOffsets[index] * BOARD_SIZE);

    MOVES::generateMoves(parentDestination, kidsDestination, isWhite);
}

/**
 * Calculates best moves for nodes based on precalculated results for their
 * sons.
 *
 * @param[out] results Pointer to array with results for each node.
 * @param boardsOffset Pointer to the array with boards for each parent node.
 * @param currentLevelBoardCount Number of boards on the considered level of
 * MIN-MAX tree (parent nodes).
 * @param lowerLevelBoardCount Number of boards on the level below considered
 * one (childreen nodes).
 */
__global__ void gatherResultsForBoards(
    int *results, unsigned int *boardsOffsets,
    unsigned int currentLevelBoardCount,
    unsigned int lowerLevelBoardCount,
    bool maximizing, int *last) {
    pos64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= currentLevelBoardCount) {
        return;
    }
    int *parentDestination = results + index;
    int *kidsDestination =
        results + currentLevelBoardCount + boardsOffsets[index];

    unsigned int kidsBoardCount = 10;

    gatherResults(parentDestination, kidsDestination, maximizing, last,
                  kidsBoardCount);
}

/**
 * Calculates evaluation for each given boards.
 *
 * @param boards Pointer to array of boards to evaluate.
 * @param boardCount Number of boards to evaluate.
 * @param[out] results Pointer to array to put calculated results.
 */
__global__ void evaluateBoards(pos64 *boards, unsigned int boardCount,
                               int *results) {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int index = blockId * blockDim.x + threadIdx.x;
    if (index >= boardCount) {
        return;
    }
    pos64 *boardAddress = boards + (index * BOARD_SIZE);
    if (boardAddress[WHITE_KING_OFFSET] == 0 &&
        boardAddress[BLACK_KING_OFFSET] == 0) {
        results[index] = INF;
    } else {
        results[index] = EVALUATION::evaluatePosition(boardAddress);
    }
}

/**
 * Calculates number of valid moves for each board.
 *
 * @param boards Pointer to array of boards.
 * @param boardsOffset Pointer to array storing offsets where boards for each
 * node in @ref boards array starts.
 * @param isWhite Wheather the considered node is on odd level. (eg. true for
 * level 1, false for level 2, and so on).
 * @param boardCount Pointer to array to put calculated results.
 */
__global__ void preCalculateBoardsCount(pos64 *boards,
                                        unsigned int *boardsOffsets,
                                        bool isWhite, int boardCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= boardCount) {
        return;
    }

    boardsOffsets[index] =
        MOVES::precountMoves(boards + (index * BOARD_SIZE), isWhite);
}

/**
 * Calculate and allocate required memory for arrays.
 *
 * @param[out] boards Pointer to place in memory where array of boards will be
 * allocated.
 * @param[out] boardsOffset Pointer to place in memory where array storing
 * offsets where boards for each node in @ref boards array starts will be
 * allocated.
 * @param[out] levelSizes Pointer to place in memory where array with number of
 * nodes on each level of MIN-MAX tree will be allocated.
 */
int prepareMemory(pos64 **boards, unsigned int **offsets,
                  unsigned int **levelSizes, int maxDepth) {
    gpuErrchk(cudaMallocManaged(levelSizes, sizeof(int) * (maxDepth + 1)));

    size_t sizeOfOneBoard = sizeof(pos64) * BOARD_SIZE + sizeof(unsigned int);
    size_t total, free;
    cudaMemGetInfo(&free, &total);

    int maxBoardCount = free / (sizeOfOneBoard + 1);
    DBG(printf("max boards: %d\n", maxBoardCount));

    gpuErrchk(cudaMalloc(boards, sizeof(pos64) * BOARD_SIZE * maxBoardCount));
    gpuErrchk(cudaMalloc(offsets, sizeof(unsigned int) * maxBoardCount));
    gpuErrchk(cudaMemset(*offsets, 0, sizeof(int) * maxBoardCount));
    gpuErrchk(
        cudaMemset(*boards, 0, sizeof(pos64) * BOARD_SIZE * maxBoardCount));

    *levelSizes[0] = 1;

    return maxBoardCount;
}

/**
 * Generates MIN-MAX tree.
 *
 * @param boards Pointer to array of boards.
 * @param boardsOffset Pointer to array storing offsets where boards for each
 * node in @ref boards array starts.
 * @param levelSizes Pointer to array storing number of nodes of each level of
 * the MIN-MAX tree.
 * @param[out] depthFound Pointer to number storing the maximum depth that was
 * reached in MIN-MAX tree.
 * @param maxBoardsCount Maximum number of boards that the generated MIN-MAX
 * tree should have.
 * @param isFirstStage Wheather it is first or second stage of the algorithm.
 */
int runBoardGeneration(pos64 *boards, unsigned int *boardsOffsets,
                       unsigned int *levelSizes, int *depthFound, bool *isWhite,
                       int maxBoardsCount, bool isFirstStage, int maxDepth) {
    int runningBoards;
    int threadCount;
    dim3 blockCount;
    int offset = 0;
    *depthFound = maxDepth;
    for (int i = 0; i < maxDepth; i++) {
        runningBoards = levelSizes[i];

        DBG(printf("generating depth: %d, running boards: %d, offset %d\n", i + 1,
                   runningBoards, offset));

        // first stage - check how many boards will be generated for each board
        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        preCalculateBoardsCount<<<blockCount, threadCount>>>(
            boards + offset * BOARD_SIZE, boardsOffsets + offset, *isWhite,
            runningBoards);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        // secound stage - find boardsOffsets for each board to put their kids
        SCAN::scan(
            boardsOffsets + offset, runningBoards,
            (unsigned int *)(boards + (offset + runningBoards) * BOARD_SIZE),
            &levelSizes[i + 1]);  // since boards are not yet created the space there
                                  // is used as a temp table

        DBG(printf("boardCount on depth %d, %u\n", i + 1, levelSizes[i + 1]));

        if ((isFirstStage &&
             levelSizes[i + 1] > MAX_BOARD_COMPUTED_IN_SECOUND_STAGE) ||
            (!isFirstStage &&
             runningBoards + offset + levelSizes[i + 1] > maxBoardsCount)) {
            *depthFound = i;
            break;
        }
        // third stage - generate boards

        DBG(printf("generate boards\n"));
        generateMovesForBoards<<<blockCount, threadCount>>>(
            boards + offset * BOARD_SIZE, boardsOffsets + offset, *isWhite,
            runningBoards);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        *isWhite = !(*isWhite);
        offset += runningBoards;
    }
    return offset;
}

/**
 * Calculate MIN-MAX tree result using standard algorithm.
 *
 * @param boards Pointer to array of boards.
 * @param boardsOffset Pointer to array storing offsets where boards for each
 * node in @ref boards array starts.
 * @param levelSizes Pointer to array storing number of nodes of each level of
 * the MIN-MAX tree.
 * @param depthFound Number storing the maximum depth that was reached in
 * MIN-MAX tree.
 * @param[out] localLast Pointer to number storing index of the best result for
 * root node, used to find best move.
 * @param offsetToLastLevel Number of nodes between root node and first node of
 * the last level of MIN-MAX tree.
 * @param maximizing Wheather considered node should maximize or minimize
 * results from sons.
 */
void gatherResults(pos64 *boards, unsigned int *boardsOffsets,
                   unsigned int *levelSizes, int depthFound, int *localLast,
                   int offsetToLastLevel, bool maximizing) {
    int threadCount, runningBoards;
    dim3 blockCount;
    int offset = offsetToLastLevel;

    for (int i = depthFound - 1; i >= 0; i--) {
        runningBoards = levelSizes[i];
        offset -= runningBoards;

        DBG(printf("running boards: %u, offset: %u\n", runningBoards, offset));
        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        std::cout << blockCount.x << " " << blockCount.y << " " << blockCount.z << " " << threadCount << std::endl;
        gatherResultsForBoards<<<blockCount, threadCount>>>(
            (int *)(boardsOffsets + offset), boardsOffsets + offset,
            runningBoards, levelSizes[i + 1], maximizing,
            localLast);  // since each thread uses the offset only once and then
                         // writes to one place the values here can just be swapped
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        maximizing = !maximizing;
    }
}

/**
 * Finds best move for given position.
 *
 * @param currentPlayer Player whose currently turn to move is ( @ref WHITE for
 * white, @ref BLACK for black)
 * @param position[in, out] Pointer to structure holding positions of pieces for 
 * which optimal move should be found. After finding the most optimal move, the
 * position is overriden with the so found move.
 */
long findBestMove(const short &currentPlayer, pos64 *position, int maxDevices, int maxDepth) {
    std::vector<std::thread> threads;
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);

    if(devicesCount > maxDevices) {
        devicesCount = maxDevices;
    }

    unsigned int *levelSizes, *boardsOffsets;
    pos64 *boards;
    size_t totalBoardsCount =
        prepareMemory(&boards, &boardsOffsets, &levelSizes, maxDepth);
    gpuErrchk(cudaMemcpy(boards, position, sizeof(pos64) * BOARD_SIZE,
                         cudaMemcpyHostToDevice));
    int firstStageDepth;
    long memoryUsage = 0;
    bool isWhite = currentPlayer == WHITE;

    if (false) {
        int offset = runBoardGeneration(boards, boardsOffsets, levelSizes,
                                        &firstStageDepth, &isWhite,
                                        totalBoardsCount, false, maxDepth);

        evaluateBoards<<<getBlocksCount2d(levelSizes[firstStageDepth]),
                         MAX_THREADS>>>(
            boards + offset * BOARD_SIZE, levelSizes[firstStageDepth],
            (int *)(boardsOffsets + offset));  // since last level doesnt use
                                               // offsets board it might be used
                                               // for keeping the evaluation
        memoryUsage = (offset + levelSizes[firstStageDepth]) * (BOARD_SIZE * sizeof(pos64) + sizeof(int));
        
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        gatherResults(boards, boardsOffsets, levelSizes, firstStageDepth, last,
                      offset, !isWhite);

        int bestMoveNr;
        gpuErrchk(
            cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(
            position, boards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE),
            sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));

        cudaFree(levelSizes);
        cudaFree(boards);
        cudaFree(boardsOffsets);
        
        return memoryUsage;
    }

    int offset =
        runBoardGeneration(boards, boardsOffsets, levelSizes, &firstStageDepth,
                           &isWhite, totalBoardsCount, true, maxDepth);

    int totalBoardCountInFirstStage = offset + levelSizes[firstStageDepth];
    pos64 *firstStageBoards =
        new pos64[totalBoardCountInFirstStage * BOARD_SIZE];
    int *firstStageOffsets = new int[totalBoardCountInFirstStage];

    gpuErrchk(
        cudaMemcpy(firstStageBoards, boards,
                   sizeof(pos64) * BOARD_SIZE * totalBoardCountInFirstStage,
                   cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(firstStageOffsets, boardsOffsets,
                         sizeof(int) * totalBoardCountInFirstStage,
                         cudaMemcpyDeviceToHost));

    cudaFree(boards);
    cudaFree(boardsOffsets);

    int boardsToCalaculateInSecStage = levelSizes[firstStageDepth];
    DBG(printf("Stage two started\n"));
    for (int j = 0; j < devicesCount; j++) {
        threads.push_back(std::thread([&, j, boardsToCalaculateInSecStage,
                                       isWhite, devicesCount]() {
            gpuErrchk(cudaSetDevice(j));

            pos64 *secStageBoards;
            unsigned int *secStageOffsets, *secStageLevelSizes;
            int *last;
            CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));

            int maxBoards = prepareMemory(&secStageBoards, &secStageOffsets,
                                          &secStageLevelSizes, maxDepth);
            pos64 *baseBoardsAddress = firstStageBoards + offset * BOARD_SIZE;
            int *baseOffsetsAddress = firstStageOffsets + offset;

            int countOfBoardsPerThread, baseCountOfBoardsPerThread;
            if (boardsToCalaculateInSecStage % devicesCount != 0) {
                if (j < devicesCount - 1) {
                    countOfBoardsPerThread = boardsToCalaculateInSecStage / devicesCount;
                    baseCountOfBoardsPerThread = countOfBoardsPerThread;
                } else {
                    countOfBoardsPerThread = boardsToCalaculateInSecStage % devicesCount;
                    baseCountOfBoardsPerThread = boardsToCalaculateInSecStage / devicesCount;
                }
            } else {
                countOfBoardsPerThread = boardsToCalaculateInSecStage / devicesCount;
                baseCountOfBoardsPerThread = countOfBoardsPerThread;
            }
            secStageLevelSizes[0] = countOfBoardsPerThread;
            gpuErrchk(cudaMemcpy(
                secStageBoards,
                baseBoardsAddress + j * baseCountOfBoardsPerThread * BOARD_SIZE,
                sizeof(pos64) * BOARD_SIZE * countOfBoardsPerThread,
                cudaMemcpyHostToDevice));
            int depthFound;
            bool isWhiteTemp = isWhite;
            int tempOffset = runBoardGeneration(secStageBoards, secStageOffsets,
                                                secStageLevelSizes, &depthFound,
                                                &isWhiteTemp, maxBoards, false, maxDepth);
            DBG(printf("offset %d, depthFound: %d\n", tempOffset, depthFound));

            // evaluating
            DBG(printf("count of evaluation boards: %u\n",
                       secStageLevelSizes[depthFound]));
            if(j == 0) {
                memoryUsage = (tempOffset + secStageLevelSizes[depthFound]) * (BOARD_SIZE * sizeof(pos64) + sizeof(int));
            }
            int threadCount;
            dim3 blockCount;
            setThreadAndBlocksCount(&threadCount, &blockCount, secStageLevelSizes[depthFound]);
        
            /*evaluateBoards<<<blockCount, threadCount>>>(
                secStageBoards + tempOffset * BOARD_SIZE,
                secStageLevelSizes[depthFound],
                (int *)(secStageOffsets +
                        tempOffset));  // since last level doesnt use
                                       // offsets board it is used
                                       // to keep the evaluation

            
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());*/

            gatherResults(secStageBoards, secStageOffsets, secStageLevelSizes,
                          depthFound, last, tempOffset, !isWhiteTemp);

            gpuErrchk(cudaMemcpy(
                baseOffsetsAddress + j * baseCountOfBoardsPerThread,
                secStageOffsets, sizeof(int) * countOfBoardsPerThread,
                cudaMemcpyDeviceToHost));
            cudaFree(secStageBoards);
            cudaFree(secStageOffsets);
            cudaFree(secStageLevelSizes);
            cudaFree(last);
        }));
    }
    for (int j = 0; j < devicesCount; j++) {
        threads[j].join();
    }
    cudaSetDevice(0);

    DBG(printf("Stage 1 - gathering results\n"));
    CHECK_ALLOC(cudaMalloc(
        &boards, sizeof(pos64) * BOARD_SIZE * totalBoardCountInFirstStage));
    CHECK_ALLOC(
        cudaMalloc(&boardsOffsets, sizeof(int) * totalBoardCountInFirstStage));

    gpuErrchk(
        cudaMemcpy(boards, firstStageBoards,
                   sizeof(pos64) * BOARD_SIZE * totalBoardCountInFirstStage,
                   cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(boardsOffsets, firstStageOffsets,
                         sizeof(int) * totalBoardCountInFirstStage,
                         cudaMemcpyHostToDevice));

    delete[] firstStageBoards;
    delete[] firstStageOffsets;

    // acquiring results for first stage
    gatherResults(boards, boardsOffsets, levelSizes, firstStageDepth, last,
                  offset, !isWhite);

    int bestMoveNr;
    gpuErrchk(
        cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(position,
                         boards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE),
                         sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));
    cudaFree(levelSizes);

    cudaFree(boards);
    cudaFree(boardsOffsets);
    return memoryUsage;
}
}  // namespace SEARCH