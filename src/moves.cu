#include "moves.cuh"

#define PAWN_OFFSET 0
#define KNIGHT_OFFSET 1
#define BISHOP_OFFSET 2
#define ROOK_OFFSET 3
#define QUEEN_OFFSET 4
#define KING_OFFSET 5

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

__host__ __device__ void copyPosition(pos64 *from, pos64 *to) {
    for(int i = 0; i < BOARD_SIZE; i++) {
        to[i] = from[i];
    }
}

__host__ __device__ void copyOneColorPieces(pos64 *from, pos64 *to) {
    for(int i = 0; i < BOARD_SIZE / 2; i++) {
        to[i] = from[i];
    }
}

__host__ __device__ void copyOneColorPiecesAndCheckIfTaken(pos64 *from, pos64* to, pos64 attack){
            for(int i = 0; i < BOARD_SIZE / 2; i++) {
        to[i] = checkIfTakenAndAssign(from[i], attack);
    }
}

__host__ __device__ bool isEmpty(pos64 *boards) {
    if(boards[WHITE_PAWN_OFFSET] == 0 && boards[WHITE_BISHOP_OFFSET] == 0 && boards[WHITE_KNIGHT_OFFSET] == 0
    && boards[WHITE_ROOK_OFFSET] == 0 && boards[WHITE_QUEEN_OFFSET] == 0 && boards[WHITE_KING_OFFSET] == 0
    && boards[BLACK_PAWN_OFFSET] == 0 && boards[BLACK_BISHOP_OFFSET] == 0 && boards[BLACK_KNIGHT_OFFSET] == 0
    && boards[BLACK_ROOK_OFFSET] == 0 && boards[BLACK_QUEEN_OFFSET] == 0 && boards[BLACK_KING_OFFSET] == 0) {
        return true;
    }
    return false;
}

 __device__ int pre_count_moves(pos64 *starting_boards, bool isWhite) 
{
    int generatedMoves = 0;

    pos64 *startingOwnPieces, *startingEnemyPieces;
    pos64 allPieces =  (starting_boards[WHITE_PAWN_OFFSET] | starting_boards[WHITE_BISHOP_OFFSET] | starting_boards[WHITE_KNIGHT_OFFSET] | starting_boards[WHITE_ROOK_OFFSET] | starting_boards[WHITE_QUEEN_OFFSET] | starting_boards[WHITE_KING_OFFSET] |
        starting_boards[BLACK_PAWN_OFFSET] | starting_boards[BLACK_BISHOP_OFFSET] | starting_boards[BLACK_KNIGHT_OFFSET] | starting_boards[BLACK_ROOK_OFFSET] | starting_boards[BLACK_QUEEN_OFFSET] | starting_boards[BLACK_KING_OFFSET]);
    pos64 enemyPieces, moves, occupied, singleMove;

    if(isWhite) {
        startingOwnPieces = starting_boards + WHITE_PAWN_OFFSET;
        startingEnemyPieces = starting_boards + BLACK_PAWN_OFFSET;
        enemyPieces = (starting_boards[BLACK_PAWN_OFFSET] | starting_boards[BLACK_BISHOP_OFFSET] | starting_boards[BLACK_KNIGHT_OFFSET] | starting_boards[BLACK_ROOK_OFFSET] | starting_boards[BLACK_QUEEN_OFFSET] | starting_boards[BLACK_KING_OFFSET]);
    } else {
        startingOwnPieces = starting_boards + BLACK_PAWN_OFFSET;
        startingEnemyPieces = starting_boards + WHITE_PAWN_OFFSET;
        enemyPieces = (starting_boards[WHITE_PAWN_OFFSET] | starting_boards[WHITE_BISHOP_OFFSET] | starting_boards[WHITE_KNIGHT_OFFSET] | starting_boards[WHITE_ROOK_OFFSET] | starting_boards[WHITE_QUEEN_OFFSET] | starting_boards[WHITE_KING_OFFSET]);
    }


    if (isWhite) { 

        //when on base position try move 2 forward
        moves = noOne(noOne((startingOwnPieces[PAWN_OFFSET] & WHITE_PAWN_STARTING_POS)));
        occupied = ((moves & allPieces) | (moves & noOne(allPieces)));
        moves = (moves ^ occupied); 
        generatedMoves += __popcll(moves);

        // generate pawn moves forward
        moves = noOne(startingOwnPieces[PAWN_OFFSET]);
        occupied = (moves & allPieces);
        moves = (moves ^ occupied);         

        generatedMoves += __popcll(moves);

        // generate pawn attacks east
        moves = (noEaOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        generatedMoves += __popcll(moves);

        // generate pawn attacks west
        moves = (noWeOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        generatedMoves += __popcll(moves);
    }
    else {

        //when on base position try move 2 forward
        moves = soOne(soOne(startingOwnPieces[PAWN_OFFSET] & BLACk_PAWN_STARTING_POS));
        occupied = ((moves & allPieces) | (moves & soOne(allPieces)));
        moves = (moves ^ occupied); 
        generatedMoves += __popcll(moves);



        // generate move forward
        moves = soOne(startingOwnPieces[PAWN_OFFSET]);
        occupied = (moves & allPieces);
        moves = (moves ^ occupied);         
        generatedMoves += __popcll(moves);

        // generate pawn attacks east
        moves = (soEaOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        generatedMoves += __popcll(moves);

        // generate pawn attacks west
        moves = (soWeOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        generatedMoves += __popcll(moves);
    }


    //knight moves
    pos64 piece;
    pos64 movingKnights = startingOwnPieces[KNIGHT_OFFSET];
    while(movingKnights != 0){
        piece = getLeastSignificantBit(movingKnights);

        moves = (noOne(noEaOne(piece)) | eastOne(noEaOne(piece)) | eastOne(soEaOne(piece)) | soOne(soEaOne(piece)) | soOne(soWeOne(piece)) | westOne(soWeOne(piece)) | westOne(noWeOne(piece)) | noOne(noWeOne(piece)));

        occupied = (moves & (allPieces ^ enemyPieces));
        moves = (moves ^ occupied);
        generatedMoves += __popcll(moves);

        movingKnights = resetLeastSignificantBit(movingKnights);
    }

    //king moves
    piece = getLeastSignificantBit(startingOwnPieces[KING_OFFSET]);
    moves = noOne(piece) | soOne(piece) | westOne(piece) | eastOne(piece) | noEaOne(piece) | noWeOne(piece) | soEaOne(piece) | soWeOne(piece);
    occupied = moves & (allPieces ^ enemyPieces);
    moves = moves ^ occupied;
    generatedMoves += __popcll(moves);

    // rooks moves
    pos64 movingRooks = startingOwnPieces[ROOK_OFFSET];
    while(movingRooks != 0){
        piece = getLeastSignificantBit(movingRooks);

        //moving north
        singleMove = piece;
        while(((singleMove = noOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving west
        singleMove = piece;
        while(((singleMove = westOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        // moving south
        singleMove = piece;
       while(((singleMove = soOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving east
        singleMove = piece;
        while(((singleMove = eastOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        movingRooks = resetLeastSignificantBit(movingRooks);
    }
    
    // bishop moves
    pos64 movingBishops = startingOwnPieces[BISHOP_OFFSET];
    while(movingBishops != 0){
        piece = getLeastSignificantBit(movingBishops);

        //moving north east
        singleMove = piece;
        while(((singleMove = noEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving north west
        singleMove = piece;
        while(((singleMove = noWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south west
        singleMove = piece;
        while(((singleMove = soWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south east
        singleMove = piece;
        while(((singleMove = soEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        movingBishops = resetLeastSignificantBit(movingBishops);
    }

    // queen moves
    pos64 movingQueens = startingOwnPieces[QUEEN_OFFSET];
    while(movingQueens != 0){
        piece = getLeastSignificantBit(movingQueens);

        //moving north
        singleMove = piece;
        while(((singleMove = noOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving west
        singleMove = piece;
        while(((singleMove = westOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        // moving south
        singleMove = piece;
       while(((singleMove = soOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving east
        singleMove = piece;
        while(((singleMove = eastOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

                singleMove = piece;
        while(((singleMove = noEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving north west
        singleMove = piece;
        while(((singleMove = noWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south west
        singleMove = piece;
        while(((singleMove = soWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south east
        singleMove = piece;
        while(((singleMove = soEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            generatedMoves++;
            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        movingQueens = resetLeastSignificantBit(movingQueens);
    }
    return generatedMoves;
}

__host__ __device__ void generate_moves(pos64 *starting_boards, pos64 * generated_boards_space, bool isWhite) {
    int generatedMoves = 0;

    pos64 *startingOwnPieces, *startingEnemyPieces;
    pos64 allPieces =  (starting_boards[WHITE_PAWN_OFFSET] | starting_boards[WHITE_BISHOP_OFFSET] | starting_boards[WHITE_KNIGHT_OFFSET] | starting_boards[WHITE_ROOK_OFFSET] | starting_boards[WHITE_QUEEN_OFFSET] | starting_boards[WHITE_KING_OFFSET] |
        starting_boards[BLACK_PAWN_OFFSET] | starting_boards[BLACK_BISHOP_OFFSET] | starting_boards[BLACK_KNIGHT_OFFSET] | starting_boards[BLACK_ROOK_OFFSET] | starting_boards[BLACK_QUEEN_OFFSET] | starting_boards[BLACK_KING_OFFSET]);
    pos64 enemyPieces, moves, occupied, singleMove;

    int currentBoardOffset = 0;
    int ownPiecesOffset, enemyPiecesOffset;
    if(isWhite) {
        ownPiecesOffset = WHITE_PAWN_OFFSET;
        enemyPiecesOffset = BLACK_PAWN_OFFSET;
        startingOwnPieces = starting_boards + WHITE_PAWN_OFFSET;
        startingEnemyPieces = starting_boards + BLACK_PAWN_OFFSET;
        enemyPieces = (starting_boards[BLACK_PAWN_OFFSET] | starting_boards[BLACK_BISHOP_OFFSET] | starting_boards[BLACK_KNIGHT_OFFSET] | starting_boards[BLACK_ROOK_OFFSET] | starting_boards[BLACK_QUEEN_OFFSET] | starting_boards[BLACK_KING_OFFSET]);
    } else {
        ownPiecesOffset = BLACK_PAWN_OFFSET;
        enemyPiecesOffset = WHITE_PAWN_OFFSET;
        startingOwnPieces = starting_boards + BLACK_PAWN_OFFSET;
        startingEnemyPieces = starting_boards + WHITE_PAWN_OFFSET;
        enemyPieces = (starting_boards[WHITE_PAWN_OFFSET] | starting_boards[WHITE_BISHOP_OFFSET] | starting_boards[WHITE_KNIGHT_OFFSET] | starting_boards[WHITE_ROOK_OFFSET] | starting_boards[WHITE_QUEEN_OFFSET] | starting_boards[WHITE_KING_OFFSET]);
    }


    if (isWhite) { 

        //when on base position try move 2 forward
        moves = noOne(noOne((startingOwnPieces[PAWN_OFFSET] & WHITE_PAWN_STARTING_POS)));
        occupied = ((moves & allPieces) | (moves & noOne(allPieces)));
        moves = (moves ^ occupied); 
        while(moves != 0) {
            copyPosition(starting_boards, generated_boards_space + currentBoardOffset);

            singleMove = getLeastSignificantBit(moves);
            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ soOne(soOne(singleMove))) | singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        // generate pawn moves forward
        moves = noOne(startingOwnPieces[PAWN_OFFSET]);
        occupied = (moves & allPieces);
        moves = (moves ^ occupied);         

        while(moves != 0) {
            copyPosition(starting_boards, generated_boards_space + currentBoardOffset);

            singleMove = getLeastSignificantBit(moves);
            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ soOne(singleMove)) | singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        // generate pawn attacks east
        moves = (noEaOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        while(moves != 0){
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            singleMove = getLeastSignificantBit(moves);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ soWeOne(singleMove)) | singleMove);

            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        // generate pawn attacks west
        moves = (noWeOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        while(moves != 0){
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            singleMove = getLeastSignificantBit(moves);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ soEaOne(singleMove)) | singleMove);

            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }
    }
    else {

        //when on base position try move 2 forward
        moves = soOne(soOne(startingOwnPieces[PAWN_OFFSET] & BLACk_PAWN_STARTING_POS));
        occupied = ((moves & allPieces) | (moves & soOne(allPieces)));
        moves = (moves ^ occupied); 
        while(moves != 0) {
            copyPosition(starting_boards, generated_boards_space + currentBoardOffset);

            singleMove = getLeastSignificantBit(moves);
            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ noOne(noOne(singleMove))) | singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }



        // generate move forward
        moves = soOne(startingOwnPieces[PAWN_OFFSET]);
        occupied = (moves & allPieces);
        moves = (moves ^ occupied);         

        while(moves != 0) {
            copyPosition(starting_boards, generated_boards_space + currentBoardOffset);

            singleMove = getLeastSignificantBit(moves);
            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ noOne(singleMove)) | singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        // generate pawn attacks east
        moves = (soEaOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        while(moves != 0){
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            singleMove = getLeastSignificantBit(moves);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ noWeOne(singleMove)) | singleMove);

            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        // generate pawn attacks west
        moves = (soWeOne(startingOwnPieces[PAWN_OFFSET]) & enemyPieces);
        while(moves != 0){
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            singleMove = getLeastSignificantBit(moves);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[PAWN_OFFSET] = ((startingOwnPieces[PAWN_OFFSET] ^ noEaOne(singleMove)) | singleMove);

            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            moves = resetLeastSignificantBit(moves);
            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }
    }


    //knight moves
    pos64 piece, attacks;
    pos64 movingKnights = startingOwnPieces[KNIGHT_OFFSET];
    while(movingKnights != 0){
        piece = getLeastSignificantBit(movingKnights);

        moves = (noOne(noEaOne(piece)) | eastOne(noEaOne(piece)) | eastOne(soEaOne(piece)) | soOne(soEaOne(piece)) | soOne(soWeOne(piece)) | westOne(soWeOne(piece)) | westOne(noWeOne(piece)) | noOne(noWeOne(piece)));

        occupied = (moves & (allPieces ^ enemyPieces));
        moves = (moves ^ occupied);
        attacks = (moves & enemyPieces);
        moves = (moves ^ attacks);
        while(moves != 0){
            copyPosition(starting_boards, generated_boards_space + currentBoardOffset);

            singleMove = getLeastSignificantBit(moves);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[KNIGHT_OFFSET] = ((startingOwnPieces[KNIGHT_OFFSET] ^ piece) | singleMove);

            moves = resetLeastSignificantBit(moves);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        while(attacks != 0){
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            singleMove = getLeastSignificantBit(attacks);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[KNIGHT_OFFSET] = (startingOwnPieces[KNIGHT_OFFSET] ^ piece) | singleMove;

            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            attacks = resetLeastSignificantBit(attacks);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

        movingKnights = resetLeastSignificantBit(movingKnights);
    }

    //king moves
    piece = getLeastSignificantBit((starting_boards + ownPiecesOffset)[KING_OFFSET]);
    moves = noOne(piece) | soOne(piece) | westOne(piece) | eastOne(piece) | noEaOne(piece) | noWeOne(piece) | soEaOne(piece) | soWeOne(piece);
    occupied = moves & (allPieces ^ enemyPieces);
    moves = moves ^ occupied;
    attacks = moves & enemyPieces;
    moves = moves ^ attacks;

    while(moves != 0){
           copyPosition(starting_boards, generated_boards_space + currentBoardOffset);

            singleMove = getLeastSignificantBit(moves);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[KING_OFFSET] = (startingOwnPieces[KING_OFFSET] ^ piece) | singleMove;

            moves = resetLeastSignificantBit(moves);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;
        }

    while(attacks != 0){
        copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

        singleMove = getLeastSignificantBit(attacks);

        (generated_boards_space + currentBoardOffset + ownPiecesOffset)[KING_OFFSET] = (startingOwnPieces[KING_OFFSET] ^ piece) | singleMove;

        copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

        attacks = resetLeastSignificantBit(attacks);

        generatedMoves++;
        currentBoardOffset += BOARD_SIZE;
    }

    // rooks moves
    pos64 movingRooks = startingOwnPieces[ROOK_OFFSET];
    while(movingRooks != 0){
        piece = getLeastSignificantBit(movingRooks);

        //moving north
        singleMove = piece;
        while(((singleMove = noOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[ROOK_OFFSET] = ((startingOwnPieces[ROOK_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving west
        singleMove = piece;
        while(((singleMove = westOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[ROOK_OFFSET] = ((startingOwnPieces[ROOK_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        // moving south
        singleMove = piece;
       while(((singleMove = soOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[ROOK_OFFSET] = ((startingOwnPieces[ROOK_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving east
        singleMove = piece;
        while(((singleMove = eastOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[ROOK_OFFSET] = ((startingOwnPieces[ROOK_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        movingRooks = resetLeastSignificantBit(movingRooks);
    }
    
    // bishop moves
    pos64 movingBishops = startingOwnPieces[BISHOP_OFFSET];
    while(movingBishops != 0){
        piece = getLeastSignificantBit(movingBishops);

        //moving north east
        singleMove = piece;
        while(((singleMove = noEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[BISHOP_OFFSET] = ((startingOwnPieces[BISHOP_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving north west
        singleMove = piece;
        while(((singleMove = noWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[BISHOP_OFFSET] = ((startingOwnPieces[BISHOP_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south west
        singleMove = piece;
        while(((singleMove = soWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[BISHOP_OFFSET] = ((startingOwnPieces[BISHOP_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south east
        singleMove = piece;
        while(((singleMove = soEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[BISHOP_OFFSET] = ((startingOwnPieces[BISHOP_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        movingBishops = resetLeastSignificantBit(movingBishops);
    }

    // queen moves
    pos64 movingQueens = startingOwnPieces[QUEEN_OFFSET];
    while(movingQueens != 0){
        piece = getLeastSignificantBit(movingQueens);

        //moving north
        singleMove = piece;
        while(((singleMove = noOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving west
        singleMove = piece;
        while(((singleMove = westOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }
        // moving south
        singleMove = piece;
       while(((singleMove = soOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving east
        singleMove = piece;
        while(((singleMove = eastOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

                singleMove = piece;
        while(((singleMove = noEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving north west
        singleMove = piece;
        while(((singleMove = noWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south west
        singleMove = piece;
        while(((singleMove = soWeOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        // moving south east
        singleMove = piece;
        while(((singleMove = soEaOne(singleMove)) != 0)) {
            if(((singleMove & allPieces) != 0) && ((singleMove & enemyPieces) == 0)){
                break;
            }
            
            copyOneColorPieces(startingOwnPieces, generated_boards_space + currentBoardOffset + ownPiecesOffset);

            (generated_boards_space + currentBoardOffset + ownPiecesOffset)[QUEEN_OFFSET] = ((startingOwnPieces[QUEEN_OFFSET] ^ piece) | singleMove);
            copyOneColorPiecesAndCheckIfTaken(startingEnemyPieces, generated_boards_space + currentBoardOffset + enemyPiecesOffset, singleMove);

            generatedMoves++;
            currentBoardOffset += BOARD_SIZE;

            if((singleMove & enemyPieces) != 0) {
                break;
            }
        }

        movingQueens = resetLeastSignificantBit(movingQueens);
    }
}