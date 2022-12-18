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

int * h_level_sizes;
int * h_subtree_sizes;
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
    printf("Zebralem wyniki i mam %d\n", result);
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

void setThreadAndBlocksCount(int * threads, int * blocks, int boardCount)
{
    if(boardCount <= MAX_THREADS) {
        *threads = boardCount;
        *blocks = 1;
    }
    else {
        *threads = MAX_THREADS;
        *blocks = (boardCount / MAX_THREADS) + 1;
    }
}

dim3 getBlocksCount2d(int boards) {
    return dim3(65535, boards % 65535 == 0 ? boards / 65535 : boards / 65535 + 1);
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
        printf("board count: %u\n", currentLevelBoardCount);
        printf("last liczba dzieci: %u, offset: %u\n", kidsBoardCount, boardsOffsets[index]);
        
    }
    else
    {
        kidsBoardCount = boardsOffsets[index + 1] - boardsOffsets[index];
        printf("liczba dzieci: %u, offsetL %u\n", kidsBoardCount, boardsOffsets[index]);
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

void search(const short& current_player,
            pos64 *position) {
    
    unsigned int *level_sizes;
    gpuErrchk(cudaMallocManaged(&level_sizes, sizeof(int) * MAX_POSSIBLE_DEPTH));
    size_t sizeOfOneBoard = sizeof(pos64) * BOARD_SIZE + sizeof(unsigned int);
    size_t total,free;
    cudaMemGetInfo(&free, &total);

    int maxBoardCount = free / sizeOfOneBoard;
    printf("max boards: %d\n", maxBoardCount);

    pos64 *boards;
    unsigned int *boardsOffsets;
    gpuErrchk(cudaMalloc(&boards,  sizeof(pos64) * BOARD_SIZE * maxBoardCount));
    gpuErrchk(cudaMalloc(&boardsOffsets, sizeof(unsigned int) * maxBoardCount));
    gpuErrchk(cudaMemset(boardsOffsets, 0, sizeof(int) * maxBoardCount));
    gpuErrchk(cudaMemset(boards, 0, sizeof(pos64) * BOARD_SIZE * maxBoardCount));

    level_sizes[0] = 1;
    cudaMemcpy(level_sizes, level_sizes, sizeof(int), cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemcpy(boards, position, sizeof(pos64) * BOARD_SIZE, cudaMemcpyHostToDevice));

    int currentBoardCount = 1;
    int runningBoards, boardOffset = 0;
    int threadCount, blockCount;
    bool isWhite = current_player == WHITE;
    int offset = 0;
    int depthFound = MAX_POSSIBLE_DEPTH;
    for(int i = 0; i < MAX_POSSIBLE_DEPTH; i++) 
    {

        runningBoards = level_sizes[i];

        printf("generating depth: %d, found boards: %d\n", i + 1, runningBoards);

        // first stage - check how many boards will be generated for each board
        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        printf("precalculate boards count\n");
        pre_calculate_boards_count<<<blockCount, threadCount>>>(boards + offset * BOARD_SIZE, boardsOffsets + offset, isWhite, runningBoards);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        // secound stage - find boardsOffsets for each board to put their kids there
        printf("calculate offsets offset: %d\n", offset);
        scan(boardsOffsets + offset, runningBoards, (unsigned int*)(boards + (offset + runningBoards) * BOARD_SIZE), &level_sizes[i + 1]); // since boards are not yet created I use the space there as a temp table

        if(level_sizes[i + 1] + offset + runningBoards > maxBoardCount) {
            depthFound = i;
            break;
        }
        // third stage - generate boards

        printf("generate boards\n");
        generate_moves_for_boards<<<blockCount, threadCount>>>(boards + offset * BOARD_SIZE, boardsOffsets + offset, isWhite, runningBoards);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        isWhite = !isWhite;
        offset += runningBoards;
    }

    // next - evaluate 
    printf("evaluation\n");
    printf("depth: %u, offset: %d\n", level_sizes[depthFound], offset);
    evaluate_boards<<<getBlocksCount2d(level_sizes[depthFound]), MAX_THREADS>>>(boards + offset * BOARD_SIZE, level_sizes[depthFound], (int*)(boardsOffsets + offset)); // since last level doesnt use offsets board i use it for keeping evaluation            
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    for (int i = depthFound - 1; i >= 0; i--) {
        printf("gathering level :%d\n", i);
        runningBoards = level_sizes[i];
        offset -= runningBoards;
        printf("%u\n", offset);

        setThreadAndBlocksCount(&threadCount, &blockCount, runningBoards);
        gather_results_for_boards<<<blockCount, threadCount>>>((int*)(boardsOffsets + offset), boardsOffsets + offset, runningBoards, level_sizes[i + 1], !isWhite, last); // since each thread uses the offset only once and then writes to one place i can just swap the values here
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        isWhite = !isWhite;
    }

    printf("reached depth %d\n", depthFound);
    
    int bestMoveNr;
    gpuErrchk(cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d\n", bestMoveNr);
    gpuErrchk(cudaMemcpy(position, boards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));
    cudaFree(boards);
    cudaFree(boardsOffsets);
    cudaFree(level_sizes);

    // prev solution   

    /*pos64 *firstStageBoards;
    int* firstStageResults;
    int firstStageTotalBoardsCount  = h_subtree_sizes[FIRST_STAGE_DEPTH];
    int secStageTotalBoardsCount = h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH];
    CHECK_ALLOC(cudaMalloc(&firstStageBoards, sizeof(pos64) * BOARD_SIZE * firstStageTotalBoardsCount));
    CHECK_ALLOC(cudaMalloc(&firstStageResults, sizeof(int) * firstStageTotalBoardsCount));

    pos64 *temp_firstStageBoards;
    int* temp_firstStageResults;

    int threadCount, blockCount;

    try {
        temp_firstStageBoards = new pos64[BOARD_SIZE * firstStageTotalBoardsCount];
        temp_firstStageResults = new int[firstStageTotalBoardsCount];
    } catch (std::bad_alloc&) {
        ERR("Operator new");
    }

    gpuErrchk(cudaMemcpy(firstStageBoards, position, BOARD_SIZE * sizeof(pos64), cudaMemcpyHostToDevice));

    // generating moves in first stage
    DBG(printf("Stage 1 - generating moves\n"));
    
    pos64 *firstBoardAddress = firstStageBoards;
    for (int i = 0; i < FIRST_STAGE_DEPTH; i++) {
        setThreadAndBlocksCount(&threadCount, &blockCount, h_level_sizes[i]);
        generate_moves_for_boards<<<blockCount, threadCount>>>(firstBoardAddress, isWhite, h_level_sizes[i]);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        firstBoardAddress = firstStageBoards + (h_subtree_sizes[i] * BOARD_SIZE);
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
        threads.push_back (std::thread ([&, j, secStageTotalBoardsCount, isWhite, devices_count] () {
            gpuErrchk(cudaSetDevice(j));

            pos64* firstBoardAddress;
            pos64 *secStageBoards;
            int *secStageResult;
            int * last;
            CHECK_ALLOC(cudaMalloc(&last, sizeof(int)));
            CHECK_ALLOC(cudaMalloc(&secStageBoards, sizeof(pos64) * BOARD_SIZE * secStageTotalBoardsCount));
            CHECK_ALLOC(cudaMalloc(&secStageResult, sizeof(int) * secStageTotalBoardsCount));
            int localThreadCount, localBlockCount;

            pos64 * basicBoardAddres = temp_firstStageBoards + (h_subtree_sizes[FIRST_STAGE_DEPTH - 1] * BOARD_SIZE); 
            int *basicResultAddress = temp_firstStageResults + h_subtree_sizes[FIRST_STAGE_DEPTH - 1];

            for (int o = j; o < h_level_sizes[FIRST_STAGE_DEPTH]; o += devices_count) {
                copy_from_cpu_to_gpu(basicBoardAddres + (o * BOARD_SIZE), secStageBoards, sizeof(pos64) * BOARD_SIZE);
                
                //generating moves
                bool isWhiteTemp = isWhite;
                firstBoardAddress = secStageBoards;
                for (int i = 0; i < MAX_DEPTH - FIRST_STAGE_DEPTH; i++) {
                    setThreadAndBlocksCount(&localThreadCount, &localBlockCount, h_level_sizes[i]);
                    generate_moves_for_boards<<<localBlockCount, localThreadCount>>>(firstBoardAddress, isWhiteTemp, h_level_sizes[i]);
                    gpuErrchk(cudaDeviceSynchronize());
                    gpuErrchk(cudaPeekAtLastError());
                    isWhiteTemp = !isWhiteTemp;
                    firstBoardAddress = secStageBoards + (h_subtree_sizes[i] * BOARD_SIZE);
                }

                // evaluating
                pos64 *lowestLevelAdress = secStageBoards + (h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH - 1] * BOARD_SIZE);
                setThreadAndBlocksCount(&localThreadCount, &localBlockCount, h_level_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH]);
                evaluate_boards<<<localBlockCount, localThreadCount>>>(lowestLevelAdress, h_level_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH], secStageResult + h_subtree_sizes[MAX_DEPTH - FIRST_STAGE_DEPTH - 1]);
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaPeekAtLastError());
                
                int *firstResultAddress;
                for (int i = MAX_DEPTH - FIRST_STAGE_DEPTH - 1; i >= 0 ; i--) {
                    if(i == 0) {
                        firstResultAddress = secStageResult;
                    }
                    else {
                        firstResultAddress = secStageResult + h_subtree_sizes[i - 1];
                    }
                    setThreadAndBlocksCount(&localThreadCount, &localBlockCount, h_level_sizes[i]);
                    gather_results_for_boards<<<localBlockCount, localThreadCount>>>(firstResultAddress, h_level_sizes[i], !isWhiteTemp, last);
                    gpuErrchk(cudaDeviceSynchronize());
                    gpuErrchk(cudaPeekAtLastError());   
                    isWhiteTemp = !isWhiteTemp;
                }

                copy_from_gpu_to_cpu(secStageResult, basicResultAddress +  o, sizeof(int));
            }
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
    int *firstResultAddress = firstStageResults;
     for (int i = FIRST_STAGE_DEPTH - 1; i >= 0; i--) {
        if(i == 0) {
            firstResultAddress = firstStageResults;
        }
        else {
            firstResultAddress = firstResultAddress + h_subtree_sizes[i - 1];
        }
        setThreadAndBlocksCount(&threadCount, &blockCount, h_level_sizes[i]);
        gather_results_for_boards<<<blockCount, threadCount>>>(firstResultAddress, h_level_sizes[i], !isWhite, last);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        isWhite = !isWhite;
    }
    DBG(printf("Stage finished successfully\n"));

    int bestMoveNr;
    gpuErrchk(cudaMemcpy(&bestMoveNr, last, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(position, firstStageBoards + BOARD_SIZE + (bestMoveNr * BOARD_SIZE), sizeof(pos64) * BOARD_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(firstStageBoards);
    cudaFree(firstStageResults);*/
}