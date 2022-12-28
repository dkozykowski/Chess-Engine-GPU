#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "evaluate.cuh"
#include "macros.cuh"
#include "moves.cuh"
#include "position.cuh"
#include "search.cuh"
#include "uci.cuh"

namespace UCI {

pos64 whitePawns;
pos64 whiteBishops;
pos64 whiteKnights;
pos64 whiteRooks;
pos64 whiteQueens;
pos64 whiteKings;
pos64 blackPawns;
pos64 blackBishops;
pos64 blackKnights;
pos64 blackRooks;
pos64 blackQueens;
pos64 blackKings;


/**
 * Restarts the game state. Restores default position and current player.
 *
 * @param currentPlayer Number indicating current player, will be set to WHITE.
 * @param moveNum Number of moves made in the game, will be set to 0.
 */
void newgame(short &currentPlayer, int &moveNum) {
    whitePawns = WHITE_PAWN_STARTING_POS;
    whiteBishops = WHITE_BISHOP_STARTING_POS;
    whiteKnights = WHITE_KNIGHT_STARTING_POS;
    whiteRooks = WHITE_ROOK_STARTING_POS;
    whiteQueens = WHITE_QUEEN_STARTING_POS;
    whiteKings = WHITE_KING_STARTING_POS;

    blackPawns = BLACK_PAWN_STARTING_POS;
    blackBishops = BLACK_BISHOP_STARTING_POS;
    blackKnights = BLACK_KNIGHT_STARTING_POS;
    blackRooks = BLACK_ROOK_STARTING_POS;
    blackQueens = BLACK_QUEEN_STARTING_POS;
    blackKings = BLACK_KING_STARTING_POS;

    currentPlayer = WHITE;
    moveNum = 0;
}

/**
 * Reads move code from console and move proper piece to given place.
 *
 * @param is Input stream from console from which the move code will be read.
 * @param[out] currentPlayer Player whose currently turn to move is ( @ref WHITE for
 * white, @ref BLACK for black).
 * @param[out] moveNum Number of moves made in the current game.
 */
void move(std::istringstream &is, short &currentPlayer, int &moveNum) {
    std::string moveToken;
    is >> std::skipws >> moveToken;

    // validate
    if (moveToken.length() != 4) {
        printf("Invalid move\n");
        return;
    }

    int fromCol = moveToken[0] >= 'a' ? moveToken[0] - 'a' : moveToken[0] - 'A';
    int fromRow = moveToken[1] - '1';
    int toCol = moveToken[2] >= 'a' ? moveToken[2] - 'a' : moveToken[2] - 'A';
    int toRow = moveToken[3] - '1';

    if (fromCol < 0 || fromRow < 0 || toCol < 0 || toRow < 0 || 8 <= fromCol ||
        8 <= fromRow || 8 <= toCol || 8 <= toRow) {
        printf("Invalid move\n");
        return;
    }

    pos64 *position[12];
    position[WHITE_PAWN_OFFSET] = &whitePawns;
    position[WHITE_BISHOP_OFFSET] = &whiteBishops;
    position[WHITE_ROOK_OFFSET] = &whiteRooks;
    position[WHITE_KNIGHT_OFFSET] = &whiteKnights;
    position[WHITE_QUEEN_OFFSET] = &whiteQueens;
    position[WHITE_KING_OFFSET] = &whiteKings;

    position[BLACK_PAWN_OFFSET] = &blackPawns;
    position[BLACK_BISHOP_OFFSET] = &blackBishops;
    position[BLACK_ROOK_OFFSET] = &blackRooks;
    position[BLACK_KNIGHT_OFFSET] = &blackKnights;
    position[BLACK_QUEEN_OFFSET] = &blackQueens;
    position[BLACK_KING_OFFSET] = &blackKings;

    POSITION::moveChess(fromCol, fromRow, toCol, toRow, currentPlayer, position);
    moveNum++;
    currentPlayer ^= 1;
}

/**
 * Prints the current state of the game to the console.
 *
 * @param currentPlayer Player whose currently turn to move is ( @ref WHITE for
 * white, @ref BLACK for black).
 * @param moveNum Number of moves made in the current game.
 */
void printGame(short currentPlayer, int moveNum) {
    printf("Move number %d\n", moveNum);
    printf("Current player - %s\n", currentPlayer == WHITE ? "White" : "Black");

    pos64 position[12];
    position[WHITE_PAWN_OFFSET] = whitePawns;
    position[WHITE_BISHOP_OFFSET] = whiteBishops;
    position[WHITE_ROOK_OFFSET] = whiteRooks;
    position[WHITE_KNIGHT_OFFSET] = whiteKnights;
    position[WHITE_QUEEN_OFFSET] = whiteQueens;
    position[WHITE_KING_OFFSET] = whiteKings;

    position[BLACK_PAWN_OFFSET] = blackPawns;
    position[BLACK_BISHOP_OFFSET] = blackBishops;
    position[BLACK_ROOK_OFFSET] = blackRooks;
    position[BLACK_KNIGHT_OFFSET] = blackKnights;
    position[BLACK_QUEEN_OFFSET] = blackQueens;
    position[BLACK_KING_OFFSET] = blackKings;
    POSITION::printPosition(position);
}

__global__ void eval(int *result, pos64 * position) {                 
    *result = EVALUATION::evaluatePosition(position);
}

void printEval() {
    int *dResult, *hResult;
    pos64 * dPosition;
    hResult = new int;

    pos64 position[12];
    position[WHITE_PAWN_OFFSET] = whitePawns;
    position[WHITE_BISHOP_OFFSET] = whiteBishops;
    position[WHITE_ROOK_OFFSET] = whiteRooks;
    position[WHITE_KNIGHT_OFFSET] = whiteKnights;
    position[WHITE_QUEEN_OFFSET] = whiteQueens;
    position[WHITE_KING_OFFSET] = whiteKings;

    position[BLACK_PAWN_OFFSET] = blackPawns;
    position[BLACK_BISHOP_OFFSET] = blackBishops;
    position[BLACK_ROOK_OFFSET] = blackRooks;
    position[BLACK_KNIGHT_OFFSET] = blackKnights;
    position[BLACK_QUEEN_OFFSET] = blackQueens;
    position[BLACK_KING_OFFSET] = blackKings;

    cudaMalloc(&dPosition, sizeof(pos64) * 12);
    cudaMalloc(&dResult, sizeof(int));
    cudaMemcpy(dPosition, position, sizeof(pos64) * 12, cudaMemcpyHostToDevice);
    eval<<<1, 1>>>(dResult, dPosition);
    cudaMemcpy(hResult, dResult, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Current evaluation from white side: %d\n", *hResult);
    delete hResult;
    cudaFree(dPosition);
    cudaFree(dResult);
}

int _log2(pos64 x) {  // asserting x is a power of two
    for (int i = 0; i < x; i++) {
        if ((x & (((pos64)1) << i)) != 0) {
            return i;
        }
    }
    return 0;
}

std::string getMoveString(pos64 currentPos, pos64 newPos) {
    pos64 diff = currentPos ^ newPos;
    pos64 from = currentPos & diff;
    pos64 to = newPos & diff;
    int from_pos = _log2(from);
    int to_pos = _log2(to);
    std::string result = "____";
    result[0] = from_pos % 8 + 'a';
    result[1] = from_pos / 8 + '1';
    result[2] = to_pos % 8 + 'a';
    result[3] = to_pos / 8 + '1';
    return result;
}

/**
 * Runs engine function searching for the best possible move and prints it to the console.
 *
 * @param currentPlayer Player whose currently turn to move is ( @ref WHITE for
 * white, @ref BLACK for black).
 * @param moveNum Number of moves made in the current game.
 */
void go(short &currentPlayer, int &moveNum) {
    pos64 *position = new pos64[12];
    position[WHITE_PAWN_OFFSET] = whitePawns;
    position[WHITE_BISHOP_OFFSET] = whiteBishops;
    position[WHITE_KNIGHT_OFFSET] = whiteKnights;
    position[WHITE_ROOK_OFFSET] = whiteRooks;
    position[WHITE_QUEEN_OFFSET] = whiteQueens;
    position[WHITE_KING_OFFSET] = whiteKings;
    position[BLACK_PAWN_OFFSET] = blackPawns;
    position[BLACK_BISHOP_OFFSET] = blackBishops;
    position[BLACK_KNIGHT_OFFSET] = blackKnights;
    position[BLACK_ROOK_OFFSET] = blackRooks;
    position[BLACK_QUEEN_OFFSET] = blackQueens;
    position[BLACK_KING_OFFSET] = blackKings;

    SEARCH::findBestMove(currentPlayer, position);

    pos64 new_whitePawns = position[WHITE_PAWN_OFFSET];
    pos64 new_whiteBishops = position[WHITE_BISHOP_OFFSET];
    pos64 new_whiteKnights = position[WHITE_KNIGHT_OFFSET];
    pos64 new_whiteRooks = position[WHITE_ROOK_OFFSET];
    pos64 new_whiteQueens = position[WHITE_QUEEN_OFFSET];
    pos64 new_whiteKings = position[WHITE_KING_OFFSET];
    pos64 new_blackPawns = position[BLACK_PAWN_OFFSET];
    pos64 new_blackBishops = position[BLACK_BISHOP_OFFSET];
    pos64 new_blackKnights = position[BLACK_KNIGHT_OFFSET];
    pos64 new_blackRooks = position[BLACK_ROOK_OFFSET];
    pos64 new_blackQueens = position[BLACK_QUEEN_OFFSET];
    pos64 new_blackKings = position[BLACK_KING_OFFSET];

    DBG2(POSITION::printPosition(position));

    if (currentPlayer == WHITE) {
        pos64 currentPos = whitePawns | whiteBishops | whiteKnights |
                           whiteRooks | whiteQueens | whiteKings;
        pos64 newPos = new_whitePawns | new_whiteBishops | new_whiteKnights |
                       new_whiteRooks | new_whiteQueens | new_whiteKings;
        std::cout << getMoveString(currentPos, newPos) << "\n";
    } else if (currentPlayer == BLACK) {
        pos64 currentPos = blackPawns | blackBishops | blackKnights |
                           blackRooks | blackQueens | blackKings;
        pos64 newPos = new_blackPawns | new_blackBishops | new_blackKnights |
                       new_blackRooks | new_blackQueens | new_blackKings;
        std::cout << getMoveString(currentPos, newPos) << "\n";
    }
}

/**
 * Prints all valid moves for current position to the console.
 *
 * @param currentPlayer Player whose currently turn to move is ( @ref WHITE for
 * white, @ref BLACK for black).
 */
void printMoves(short currentPlayer) {
    pos64 position[12];
    position[WHITE_PAWN_OFFSET] = whitePawns;
    position[WHITE_BISHOP_OFFSET] = whiteBishops;
    position[WHITE_ROOK_OFFSET] = whiteRooks;
    position[WHITE_KNIGHT_OFFSET] = whiteKnights;
    position[WHITE_QUEEN_OFFSET] = whiteQueens;
    position[WHITE_KING_OFFSET] = whiteKings;

    position[BLACK_PAWN_OFFSET] = blackPawns;
    position[BLACK_BISHOP_OFFSET] = blackBishops;
    position[BLACK_ROOK_OFFSET] = blackRooks;
    position[BLACK_KNIGHT_OFFSET] = blackKnights;
    position[BLACK_QUEEN_OFFSET] = blackQueens;
    position[BLACK_KING_OFFSET] = blackKings;

    pos64 *generatedBoards = new pos64[255 * BOARD_SIZE];

    MOVES::generateMoves(position, generatedBoards, currentPlayer == WHITE);
    std::string any;
    for (int x = 0; x < 255; x++) {
        if (((generatedBoards + (x * BOARD_SIZE))[BLACK_KING_OFFSET] |
             (generatedBoards + (x * BOARD_SIZE))[WHITE_KING_OFFSET]) == 0)
            break;
        POSITION::printPosition(generatedBoards + (x * BOARD_SIZE));

        std::getline(std::cin, any);
        if (any == "q") break;
    }
    free(generatedBoards);
}

/**
 * Main engine loop. Waits for commands and runs adequate functions.
 */
void loop() {
    short currentPlayer;
    int moveNum;
    newgame(currentPlayer, moveNum);
    SEARCH::init();

    std::string token, cmd;

    do {
        if (!std::getline(std::cin, cmd)) break;

        std::istringstream is(cmd);

        token
            .clear();  // Avoid a stale if getline() returns empty or blank line
        is >> std::skipws >> token;

        if (token == "exit" || token == "quit" || token == "stop" ||
            token == "q")
            break;
        else if (token == "ucinewgame")
            newgame(currentPlayer, moveNum);
        else if (token == "d")
            printGame(currentPlayer, moveNum);
        else if (token == "move")
            move(is, currentPlayer, moveNum);
        else if (token == "go")
            go(currentPlayer, moveNum);
        // else if (token == "bench")      bench(pos, is, states);
        else if (token == "eval")
            printEval();
        else if (token == "moves")
            printMoves(currentPlayer);
        else
            std::cout << "Unknown command: " << cmd << std::endl;
    } while (true);

    SEARCH::terminate();
}
}  // namespace UCI