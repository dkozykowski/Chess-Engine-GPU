#include <algorithm>

#include "evaluate.cuh"
#include "macros.cuh"
#include "moves.cuh"
#include "scan.cuh"
#include "search.cuh"
#include "thread"
#include "vector"

namespace SEARCH {

int *last;

__device__ void gatherResults(int *resultsTo, int *resultsFrom, bool maximize,
                              int *last, int boardCount) {
    int result;
    if (maximize) {  // maximizing
        result = -INF;
        for (int i = 0; i < boardCount; i++) {
            if (resultsFrom[i] != INF && resultsFrom[i] != -INF &&
                resultsFrom[i] > result) {
                result = resultsFrom[i];
                *last = i;
            }
        }
    } else {  // minimizing
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

void init() {
    cudaSetDevice(0);
    CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));
}

void terminate() {
    cudaSetDevice(0);
    cudaFree(last);
}

dim3 getBlocksCount2d(int boards) {
    return dim3(MAX_BLOCKS_PER_DIMENSION,
                boards % MAX_BLOCKS_PER_DIMENSION == 0
                    ? boards / MAX_BLOCKS_PER_DIMENSION
                    : boards / MAX_BLOCKS_PER_DIMENSION + 1);
}

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

__global__ void gatherResultsForBoards(
    int *results, unsigned int *boardsOffsets,
    unsigned int currentLevelBoardCount,
    unsigned int lowerLevelBoardCount,  // count of lower level
    bool maximizing, int *last) {
    pos64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= currentLevelBoardCount) {
        return;
    }
    int *parentDestination = results + index;
    int *kidsDestination =
        results + currentLevelBoardCount + boardsOffsets[index];

    unsigned int kidsBoardCount;
    if (index == (currentLevelBoardCount - 1)) {
        kidsBoardCount = lowerLevelBoardCount - boardsOffsets[index];
    } else {
        kidsBoardCount = boardsOffsets[index + 1] - boardsOffsets[index];
    }

    gatherResults(parentDestination, kidsDestination, maximizing, last,
                  kidsBoardCount);
}

__global__ void evaluateBoards(pos64 *boards, unsigned int boardCount,
                               int *results) {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int index = blockId * blockDim.x + threadIdx.x;
    if (index >= boardCount) {
        return;
    }
    pos64 *boardAddress = boards + (index * BOARD_SIZE);
    if (boardAddress[WHITE_KING_OFFSET] == 0 &&
        boardAddress[BLACK_KING_OFFSET] == 0) {  // is it properly handled ??
        results[index] = INF;
    } else {
        results[index] = EVALUATION::evaluatePosition(
            boardAddress[WHITE_PAWN_OFFSET], boardAddress[WHITE_BISHOP_OFFSET],
            boardAddress[WHITE_KNIGHT_OFFSET], boardAddress[WHITE_ROOK_OFFSET],
            boardAddress[WHITE_QUEEN_OFFSET], boardAddress[WHITE_KING_OFFSET],
            boardAddress[BLACK_PAWN_OFFSET], boardAddress[BLACK_BISHOP_OFFSET],
            boardAddress[BLACK_KNIGHT_OFFSET], boardAddress[BLACK_ROOK_OFFSET],
            boardAddress[BLACK_QUEEN_OFFSET], boardAddress[BLACK_KING_OFFSET]);
    }
}

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

int prepareMemory(pos64 **boards, unsigned int **offsets,
                  unsigned int **levelSizes) {
    gpuErrchk(cudaMallocManaged(levelSizes, sizeof(int) * MAX_POSSIBLE_DEPTH));

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

int runBoardGeneration(pos64 *boards, unsigned int *boardsOffsets,
                       unsigned int *levelSizes, int *depthFound, bool *isWhite,
                       int maxBoardsCount, bool isFirstStage) {
    int runningBoards;
    int threadCount;
    dim3 blockCount;
    int offset = 0;
    *depthFound = MAX_POSSIBLE_DEPTH;
    for (int i = 0; i < MAX_POSSIBLE_DEPTH; i++) {
        runningBoards = levelSizes[i];

        DBG(printf("generating depth: %d, evaluated boards: %d\n", i + 1,
                   runningBoards));

        // first stage - check how many boards will be generated for each board
        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        preCalculateBoardsCount<<<blockCount, threadCount>>>(
            boards + offset * BOARD_SIZE, boardsOffsets + offset, *isWhite,
            runningBoards);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        // secound stage - find boardsOffsets for each board to put their kids
        // there
        DBG(printf("calculate offsets offset: %d\n", offset));
        SCAN::scan(boardsOffsets + offset, runningBoards,
             (unsigned int *)(boards + (offset + runningBoards) * BOARD_SIZE),
             &levelSizes[i + 1]);  // since boards are not yet created I use the
                                   // space there as a temp table

        DBG(printf("boardCount on depth %d, %u\n", i, levelSizes[i + 1]));

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

void gatherResults(pos64 *boards, unsigned int *boardsOffsets,
                   unsigned int *levelSizes, int depthFound, int *localLast,
                   int offsetToLastLevel, bool maximizing) {
    int threadCount, runningBoards;
    dim3 blockCount;
    int offset = offsetToLastLevel;

    for (int i = depthFound - 1; i >= 0; i--) {
        runningBoards = levelSizes[i];
        offset -= runningBoards;

        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        gatherResultsForBoards<<<blockCount, threadCount>>>(
            (int *)(boardsOffsets + offset), boardsOffsets + offset,
            runningBoards, levelSizes[i + 1], maximizing,
            localLast);  // since each thread uses the offset only once and then
                         // writes to one place i can just swap the values here
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        maximizing = !maximizing;
    }
}

void findBestMove(const short &currentPlayer, pos64 *position) {
    std::vector<std::thread> threads;
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);

    unsigned int *levelSizes, *boardsOffsets;
    pos64 *boards;
    size_t totalBoardsCount =
        prepareMemory(&boards, &boardsOffsets, &levelSizes);
    gpuErrchk(cudaMemcpy(boards, position, sizeof(pos64) * BOARD_SIZE,
                         cudaMemcpyHostToDevice));
    int firstStageDepth;
    bool isWhite = currentPlayer == WHITE;

    if (devicesCount == 1) {
        int offset = runBoardGeneration(boards, boardsOffsets, levelSizes,
                                        &firstStageDepth, &isWhite,
                                        totalBoardsCount, false);

        evaluateBoards<<<getBlocksCount2d(levelSizes[firstStageDepth]),
                         MAX_THREADS>>>(
            boards + offset * BOARD_SIZE, levelSizes[firstStageDepth],
            (int *)(boardsOffsets + offset));  // since last level doesnt use
                                               // offsets board i use it for
                                               // keeping evaluation
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
        return;
    }

    int offset =
        runBoardGeneration(boards, boardsOffsets, levelSizes, &firstStageDepth,
                           &isWhite, totalBoardsCount, true);

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
                                          &secStageLevelSizes);
            pos64 *baseBoardsAddress = firstStageBoards + offset * BOARD_SIZE;
            int *baseOffsetsAddress = firstStageOffsets + offset;

            int countOfBoardsPerThread, baseCountOfBoardsPerThread;
            if (boardsToCalaculateInSecStage % devicesCount != 0) {
                if (j < devicesCount - 1) {
                    countOfBoardsPerThread =
                        boardsToCalaculateInSecStage / devicesCount + 1;
                    baseCountOfBoardsPerThread = countOfBoardsPerThread;
                } else {
                    countOfBoardsPerThread =
                        boardsToCalaculateInSecStage % devicesCount;
                    baseCountOfBoardsPerThread =
                        boardsToCalaculateInSecStage / devicesCount + 1;
                }
            } else {
                countOfBoardsPerThread =
                    boardsToCalaculateInSecStage / devicesCount;
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
                                                &isWhiteTemp, maxBoards, false);
            DBG(printf("offset %d, depthFound: %d\n", tempOffset, depthFound));

            // evaluating
            DBG(printf("count of evaluation boards: %u\n",
                       secStageLevelSizes[depthFound]));
            evaluateBoards<<<getBlocksCount2d(secStageLevelSizes[depthFound]),
                             MAX_THREADS>>>(
                boards + tempOffset * BOARD_SIZE,
                secStageLevelSizes[depthFound],
                (int *)(boardsOffsets +
                        tempOffset));  // since last level doesnt use
                                       // offsets board i use it for
                                       // keeping evaluation
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());

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
}
} // namespace SEARCH