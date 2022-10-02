#include "moves.cuh"

__host__ __device__ pos64 eastOne (pos64 position){
    return ((position << 1) & NOT_A_FILE);
}

__host__ __device__ pos64 noEaOne (pos64 position){
    return (position << 9) & NOT_A_FILE;
}

__host__ __device__ pos64 soEaOne (pos64 position){
    return (position >> 7) & NOT_A_FILE;
}

__host__ __device__ pos64 westOne (pos64 position){
    return (position >> 1) & NOT_H_FILE;
}

__host__ __device__ pos64 soWeOne (pos64 position){
    return (position >> 9) & NOT_H_FILE;
}

__host__ __device__ pos64 noWeOne (pos64 position){
    return (position << 7) & NOT_H_FILE;
}

__host__ __device__ pos64 noOne (pos64 position){
    return (position << 8);
}

__host__ __device__ pos64 soOne (pos64 position){
    return (position >> 8);
}

__host__ __device__ pos64 getLeastSignificantBit(pos64 x){
    return x & -x;
}

__host__ __device__ pos64 resetLeastSignificantBit(pos64 x){
    return x & (x - 1);
}

__host__ __device__ pos64 checkIfTakenAndAssign(pos64 pieces, pos64 attack) {
    return pieces ^ (attack & pieces);
}

__host__ __device__ void generate_moves(pos64 * start_white_pawns_boards,
                    pos64 * start_white_bishops_boards,
                    pos64 * start_white_knights_boards,
                    pos64 * start_white_rooks_boards,
                    pos64 * start_white_queens_boards,
                    pos64 * start_white_kings_boards,
                    pos64 * start_black_pawns_boards,
                    pos64 * start_black_bishops_boards,
                    pos64 * start_black_knights_boards,
                    pos64 * start_black_rooks_boards,
                    pos64 * start_black_queens_boards,
                    pos64 * start_black_kings_boards,
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
                    int * results,
                    int * depths,
                    short * stack_states,
                    int depth) {
    // zakladamy, ze korzeniem drzewa, czyli graczem dla którego szukamu ruchu, jest gracz CZARNY
    // zakladamy, ze korzen ma numer 0

    pos64 initialOwnPawns, initialOwnBishops, initialOwnKnights, initialOwnRooks, initialOwnQueens, initialOwnKings, initialEnemyPawns, initialEnemyBishops, initialEnemyKnights, initialEnemyRooks, initialEnemyQueens, initialEnemyKings;
    pos64 *ownKnights, *ownRooks, *ownQueens, *ownKings, *ownBishops, *ownPawns, *enemyKnights, *enemyRooks, *enemyQueens, *enemyKings, *enemyBishops, *enemyPawns;
    pos64 allPieces =  *start_white_pawns_boards | *start_white_bishops_boards | *start_white_knights_boards | *start_white_rooks_boards | *start_white_queens_boards | *start_white_kings_boards |
        *start_black_pawns_boards | *start_black_bishops_boards | *start_black_knights_boards | *start_black_rooks_boards | *start_black_queens_boards | *start_black_kings_boards;

    pos64 enemyPieces, moves, occupied, singleMove;
    int generatedMoves = 0;
    if ((depth & 1) == 1) { // white pawn moves
        enemyPieces = *start_black_pawns_boards | *start_black_bishops_boards | *start_black_knights_boards | *start_black_rooks_boards | *start_black_queens_boards | *start_black_kings_boards;
        
        initialOwnPawns = *start_white_pawns_boards;
        initialOwnBishops = *start_white_bishops_boards;
        initialOwnKnights = *start_white_knights_boards;
        initialOwnRooks = *start_white_rooks_boards;
        initialOwnQueens  = *start_white_queens_boards;
        initialOwnKings  = *start_white_kings_boards;
        initialEnemyPawns = *start_black_pawns_boards;
        initialEnemyBishops = *start_black_bishops_boards;
        initialEnemyKnights = *start_black_knights_boards;
        initialEnemyRooks = *start_black_rooks_boards;
        initialEnemyQueens  = *start_black_queens_boards;
        initialEnemyKings  = *start_black_kings_boards;

        ownKnights = white_knights_boards;
        ownRooks = white_rooks_boards;
        ownQueens = white_queens_boards;
        ownKings = white_kings_boards;
        ownBishops = white_bishops_boards;
        ownPawns = white_pawns_boards;
        enemyKnights = black_knights_boards;
        enemyRooks = black_rooks_boards;
        enemyQueens = black_queens_boards;
        enemyKings = black_kings_boards;
        enemyBishops = black_bishops_boards;
        enemyPawns = black_bishops_boards;

        // generate pawn moves forward
        moves = noOne(initialOwnPawns);
        occupied = moves & allPieces;
        moves = moves ^ occupied;         

        while(moves != 0 && generatedMoves < BOARDS_GENERATED) {
            black_pawns_boards[generatedMoves] = initialEnemyPawns;
            black_bishops_boards[generatedMoves] = initialEnemyBishops;
            black_knights_boards[generatedMoves] = initialEnemyKnights;
            black_rooks_boards[generatedMoves] = initialEnemyRooks;
            black_queens_boards[generatedMoves] = initialEnemyQueens;
            black_kings_boards[generatedMoves] = initialEnemyKings;
            white_bishops_boards[generatedMoves] = initialOwnBishops;
            white_knights_boards[generatedMoves] = initialOwnKnights;
            white_rooks_boards[generatedMoves] = initialOwnRooks;
            white_queens_boards[generatedMoves] = initialOwnQueens;
            white_kings_boards[generatedMoves] = initialOwnKings;

            singleMove = getLeastSignificantBit(moves);
            white_pawns_boards[generatedMoves] = (initialOwnPawns ^ soOne(singleMove)) | singleMove;

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
        }

        // generate pawn attacks east
        moves = noEaOne(initialOwnPawns) & enemyPieces;
        while(moves != 0 && generatedMoves < BOARDS_GENERATED){
            white_bishops_boards[generatedMoves] = initialOwnBishops;
            white_knights_boards[generatedMoves] = initialOwnKnights;
            white_rooks_boards[generatedMoves] = initialOwnRooks;
            white_queens_boards[generatedMoves] = initialOwnQueens;
            white_kings_boards[generatedMoves] = initialOwnKings;

            singleMove = getLeastSignificantBit(moves);

            white_pawns_boards[generatedMoves] = (initialOwnPawns ^ soWeOne(singleMove)) | singleMove;

            // finding a piece that has been taken
            black_pawns_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            black_bishops_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            black_knights_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            black_rooks_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            black_queens_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            black_kings_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
        }

        // generate pawn attacks west
        moves = noWeOne(white_pawns_boards[0]) & enemyPieces;
        while(moves != 0 && generatedMoves < BOARDS_GENERATED){
            white_bishops_boards[generatedMoves] = initialOwnBishops;
            white_knights_boards[generatedMoves] = initialOwnKnights;
            white_rooks_boards[generatedMoves] = initialOwnRooks;
            white_queens_boards[generatedMoves] = initialOwnQueens;
            white_kings_boards[generatedMoves] = initialOwnKings;

            singleMove = getLeastSignificantBit(moves);

            white_pawns_boards[generatedMoves] = (initialOwnPawns ^ soEaOne(singleMove)) | singleMove;

            // finding a piece that has been taken
            black_pawns_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            black_bishops_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            black_knights_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            black_rooks_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            black_queens_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            black_kings_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
        }
    }
    else
    {
        enemyPieces = white_pawns_boards[0] | white_bishops_boards[0] | white_knights_boards[0] | white_rooks_boards[0] | white_queens_boards[0] | white_kings_boards[0];
        
        initialOwnPawns = black_pawns_boards[0];
        initialOwnBishops = black_bishops_boards[0];
        initialOwnKnights = black_knights_boards[0];
        initialOwnRooks = black_rooks_boards[0];
        initialOwnQueens  = black_queens_boards[0];
        initialOwnKings  = black_kings_boards[0];
        initialEnemyPawns = white_pawns_boards[0];
        initialEnemyBishops = white_bishops_boards[0];
        initialEnemyKnights = white_knights_boards[0];
        initialEnemyRooks = white_rooks_boards[0];
        initialEnemyQueens  = white_queens_boards[0];
        initialEnemyKings  = white_kings_boards[0];

        ownKnights = black_knights_boards;
        ownRooks = black_rooks_boards;
        ownQueens = black_queens_boards;
        ownKings = black_kings_boards;
        ownBishops = black_bishops_boards;
        ownPawns = black_pawns_boards;
        enemyKnights = white_knights_boards;
        enemyRooks = white_rooks_boards;
        enemyQueens = white_queens_boards;
        enemyKings = white_kings_boards;
        enemyBishops = white_bishops_boards;
        enemyPawns = white_bishops_boards;
        
        // generate pawn moves forward
        moves = soOne(initialOwnPawns);
        occupied = moves & allPieces;
        moves = moves ^ occupied;         

        while(moves != 0 && generatedMoves < BOARDS_GENERATED) {
            black_bishops_boards[generatedMoves] = initialOwnBishops;
            black_knights_boards[generatedMoves] = initialOwnKnights;
            black_rooks_boards[generatedMoves] = initialOwnRooks;
            black_queens_boards[generatedMoves] = initialOwnQueens;
            black_kings_boards[generatedMoves] = initialOwnKings;
            white_bishops_boards[generatedMoves] = initialEnemyBishops;
            white_knights_boards[generatedMoves] = initialEnemyKnights;
            white_rooks_boards[generatedMoves] = initialEnemyRooks;
            white_queens_boards[generatedMoves] = initialEnemyQueens;
            white_kings_boards[generatedMoves] = initialEnemyKings;
            white_pawns_boards[generatedMoves] = initialEnemyPawns;

            singleMove = getLeastSignificantBit(moves);
            black_pawns_boards[generatedMoves] = (initialOwnPawns ^ noOne(singleMove)) | singleMove;

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
        }

        // generate pawn attacks east
        moves = soEaOne(initialOwnPawns) & enemyPieces;
        while(moves != 0 && generatedMoves < BOARDS_GENERATED){
            black_bishops_boards[generatedMoves] = initialOwnBishops;
            black_knights_boards[generatedMoves] = initialOwnKnights;
            black_rooks_boards[generatedMoves] = initialOwnRooks;
            black_queens_boards[generatedMoves] = initialOwnQueens;
            black_kings_boards[generatedMoves] = initialOwnKings;

            singleMove = getLeastSignificantBit(moves);

            black_pawns_boards[generatedMoves] = (initialOwnPawns ^ noWeOne(singleMove)) | singleMove;

            // finding a piece that has been taken
            white_pawns_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            white_bishops_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            white_knights_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            white_rooks_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            white_queens_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            white_kings_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
        }

        // generate pawn attacks west
        moves = soWeOne(white_pawns_boards[0]) & enemyPieces;
        while(moves != 0 && generatedMoves < BOARDS_GENERATED){
            black_bishops_boards[generatedMoves] = initialOwnBishops;
            black_knights_boards[generatedMoves] = initialOwnKnights;
            black_rooks_boards[generatedMoves] = initialOwnRooks;
            black_queens_boards[generatedMoves] = initialOwnQueens;
            black_kings_boards[generatedMoves] = initialOwnKings;

            singleMove = getLeastSignificantBit(moves);

            black_pawns_boards[generatedMoves] = (initialOwnPawns ^ noEaOne(singleMove)) | singleMove;

            // finding a piece that has been taken
            white_pawns_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            white_bishops_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            white_knights_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            white_rooks_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            white_queens_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            white_kings_boards[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
        }
    }


    //knight moves
    pos64 piece, attacks;
    pos64 movingKnights = initialOwnKnights;
    while(movingKnights != 0){
        piece = getLeastSignificantBit(movingKnights);

        moves = (noOne(noEaOne(piece)) | eastOne(noEaOne(piece)) | eastOne(soEaOne(piece)) | soOne(soEaOne(piece)) | soOne(soWeOne(piece)) | westOne(soWeOne(piece)) | westOne(noWeOne(piece)) | noOne(noWeOne(piece)));
        occupied = moves & (allPieces ^ enemyPieces);
        moves = moves ^ occupied;
        attacks = moves & enemyPieces;
        moves = moves ^ attacks;

        while(moves != 0 && generatedMoves < BOARDS_GENERATED){
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKings[generatedMoves] = initialOwnKings;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownRooks[generatedMoves] = initialOwnRooks;
            enemyBishops[generatedMoves] = initialEnemyBishops;
            enemyKings[generatedMoves] = initialEnemyKings;
            enemyPawns[generatedMoves] = initialEnemyPawns;
            enemyKnights[generatedMoves] = initialEnemyKnights;
            enemyQueens[generatedMoves] = initialEnemyQueens;
            enemyRooks[generatedMoves] = initialEnemyRooks;

            singleMove = getLeastSignificantBit(moves);

            ownKnights[generatedMoves] = (initialOwnKnights ^ piece) | singleMove;

            moves = resetLeastSignificantBit(moves);

            generatedMoves++;
        }

        while(attacks != 0 && generatedMoves < BOARDS_GENERATED){
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKings[generatedMoves] = initialOwnKings;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownRooks[generatedMoves] = initialOwnRooks;

            singleMove = getLeastSignificantBit(moves);

            ownKnights[generatedMoves] = (initialOwnKnights ^ piece) | singleMove;

            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);

            moves = resetLeastSignificantBit(moves);

            generatedMoves++;
        }

        movingKnights = resetLeastSignificantBit(movingKnights);
    }

    //king moves
    piece = getLeastSignificantBit(initialOwnKings);
    moves = noOne(piece) | soOne(piece) | westOne(piece) | eastOne(piece) | noEaOne(piece) | noWeOne(piece) | soEaOne(piece) | soWeOne(piece);
    occupied = moves & (allPieces ^ enemyPieces);
    moves = moves ^ occupied;
    attacks = moves & enemyPieces;
    moves = moves ^ attacks;

    while(moves != 0 && generatedMoves < BOARDS_GENERATED){
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownRooks[generatedMoves] = initialOwnRooks;
            enemyBishops[generatedMoves] = initialEnemyBishops;
            enemyKings[generatedMoves] = initialEnemyKings;
            enemyPawns[generatedMoves] = initialEnemyPawns;
            enemyKnights[generatedMoves] = initialEnemyKnights;
            enemyQueens[generatedMoves] = initialEnemyQueens;
            enemyRooks[generatedMoves] = initialEnemyRooks;

            singleMove = getLeastSignificantBit(moves);

            ownKings[generatedMoves] = (initialOwnKings ^ piece) | singleMove;

            moves = resetLeastSignificantBit(moves);

            generatedMoves++;
        }

    while(attacks != 0 && generatedMoves < BOARDS_GENERATED){
        ownBishops[generatedMoves] = initialOwnBishops;
        ownKnights[generatedMoves] = initialOwnKnights;
        ownPawns[generatedMoves] = initialOwnPawns;
        ownQueens[generatedMoves] = initialOwnQueens;
        ownRooks[generatedMoves] = initialOwnRooks;

        singleMove = getLeastSignificantBit(moves);

        ownKings[generatedMoves] = (initialOwnKings ^ piece) | singleMove;

        enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
        enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
        enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
        enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
        enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
        enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);

        moves = resetLeastSignificantBit(moves);

        generatedMoves++;
    }

    // rooks moves
    pos64 movingRooks = initialOwnRooks;
    while(movingRooks != 0 && generatedMoves < BOARDS_GENERATED){
        piece = getLeastSignificantBit(movingRooks);

        //moving north
        singleMove = piece;
        while((singleMove = noOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownRooks[generatedMoves] = (initialOwnRooks ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }

        // moving west
        singleMove = piece;
        while((singleMove = westOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownRooks[generatedMoves] = (initialOwnRooks ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }

        // moving south
        singleMove = piece;
        while((singleMove = soOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownRooks[generatedMoves] = (initialOwnRooks ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }

        // moving east
        singleMove = piece;
        while((singleMove = eastOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownRooks[generatedMoves] = (initialOwnRooks ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }
        movingRooks = resetLeastSignificantBit(movingRooks);
    }

    // bishop moves
    pos64 movingBishops = initialOwnBishops;
    while(movingBishops != 0 && generatedMoves < BOARDS_GENERATED){
        piece = getLeastSignificantBit(movingBishops);

        //moving north east
        singleMove = piece;
        while((singleMove = noEaOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownBishops[generatedMoves] = initialOwnBishops;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownRooks[generatedMoves] = (initialOwnRooks ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }

        // moving north west
        singleMove = piece;
        while((singleMove = noWeOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownRooks[generatedMoves] = initialOwnRooks;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownBishops[generatedMoves] = (initialOwnBishops ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }

        // moving south west
        singleMove = piece;
        while((singleMove = soWeOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownRooks[generatedMoves] = initialOwnRooks;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownBishops[generatedMoves] = (initialOwnBishops ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }

        // moving south east
        singleMove = piece;
        while((singleMove = soEaOne(singleMove) > piece) && generatedMoves < BOARDS_GENERATED) {
            if(singleMove == 0){
                break;
            }
            if((singleMove & allPieces != 0) && (singleMove & enemyPieces == 0)){
                break;
            }
            ownRooks[generatedMoves] = initialOwnRooks;
            ownKnights[generatedMoves] = initialOwnKnights;
            ownPawns[generatedMoves] = initialOwnPawns;
            ownQueens[generatedMoves] = initialOwnQueens;
            ownKings[generatedMoves] = initialOwnKings;
            ownBishops[generatedMoves] = (initialOwnBishops ^ piece) | singleMove;
            enemyBishops[generatedMoves] = checkIfTakenAndAssign(initialEnemyBishops, singleMove);
            enemyKings[generatedMoves] = checkIfTakenAndAssign(initialEnemyKings, singleMove);
            enemyPawns[generatedMoves] = checkIfTakenAndAssign(initialEnemyPawns, singleMove);
            enemyKnights[generatedMoves] = checkIfTakenAndAssign(initialEnemyKnights, singleMove);
            enemyQueens[generatedMoves] = checkIfTakenAndAssign(initialEnemyQueens, singleMove);
            enemyRooks[generatedMoves] = checkIfTakenAndAssign(initialEnemyRooks, singleMove);
            generatedMoves++;
        }
        movingBishops = resetLeastSignificantBit(movingRooks);
    }

    for(int wsk = 0; wsk < BOARDS_GENERATED; wsk++)
    {
        depths[wsk] = depth;
        stack_states[wsk] = RIGHT;
        if ((depth & 1) == 0) {
            results[wsk] = -INF;
        }
        else {
            results[wsk] = INF;
        }
    }

    for(int i = generatedMoves; i < BOARDS_GENERATED; i++)
    {
        black_pawns_boards[i] = 0;
        black_bishops_boards[i] = 0;
        black_knights_boards[i] = 0;
        black_rooks_boards[i] = 0;
        black_queens_boards[i] = 0;
        black_kings_boards[i] = 0;
        white_bishops_boards[i] = 0;
        white_knights_boards[i] = 0;
        white_rooks_boards[i] = 0;
        white_queens_boards[i] = 0;
        white_kings_boards[i] = 0;
        white_pawns_boards[i] = 0;
    }

}