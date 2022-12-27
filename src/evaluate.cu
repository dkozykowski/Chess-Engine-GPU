#include "evaluate.cuh"
#include "macros.cuh"

namespace EVALUATION {

// piece_square tables
__device__ int mgPawnTable[64] = {
     82,  82,  82,  82,  82,  82,  82, 82, 
    180, 216, 143, 177, 150, 208, 116, 71, 
     76,  89, 108, 113, 147, 138, 107, 62, 
     68,  95,  88, 103, 105,  94,  99, 59, 
     55,  80,  77,  94,  99,  88,  92, 57, 
     56,  78,  78,  72,  85,  85, 115, 70, 
     47,  81,  62,  59,  67, 106, 120, 60, 
     82,  82,  82,  82,  82,  82,  82, 82
};

__device__ int egPawnTable[64] = {
     94,  94,  94, 94,  94,  94,  94,  94, 
    272, 267, 252, 228, 241, 226, 259, 281, 
    188, 194, 179, 161, 150, 147, 176, 178, 
    126, 118, 107,  99,  92,  98, 111, 111, 
    107, 103,  91,  87,  87,  86,  97,  93, 
     98, 101,  88,  95,  94,  89,  93,  86, 
    107, 102, 102, 104, 107,  94,  96,  87, 
     94,  94,  94,  94,  94,  94,  94,  94
};

__device__ int mgKnightTable[64] = {
    170, 248, 303, 288, 398, 240, 322, 230, 
    264, 296, 409, 373, 360, 399, 344, 320, 
    290, 397, 374, 402, 421, 466, 410, 381, 
    328, 354, 356, 390, 374, 406, 355, 359, 
    324, 341, 353, 350, 365, 356, 358, 329, 
    314, 328, 349, 347, 356, 354, 362, 321, 
    308, 284, 325, 334, 336, 355, 323, 318, 
    232, 316, 279, 304, 320, 309, 318, 314
};

__device__ int egKnightTable[64] = {
    223, 243, 268, 253, 250, 254, 218, 182, 
    256, 273, 256, 279, 272, 256, 257, 229, 
    257, 261, 291, 290, 280, 272, 262, 240, 
    264, 284, 303, 303, 303, 292, 289, 263, 
    263, 275, 297, 306, 297, 298, 285, 263, 
    258, 278, 280, 296, 291, 278, 261, 259,
    239, 261, 271, 276, 279, 261, 258, 237, 
    252, 230, 258, 266, 259, 263, 231, 217
};

__device__ int mgBishopTable[64] = {
    336, 369, 283, 328, 340, 323, 372, 357, 
    339, 381, 347, 352, 395, 424, 383, 318, 
    349, 402, 408, 405, 400, 415, 402, 363, 
    361, 370, 384, 415, 402, 402, 372, 363, 
    359, 378, 378, 391, 399, 377, 375, 369, 
    365, 380, 380, 380, 379, 392, 383, 375, 
    369, 380, 381, 365, 372, 386, 398, 366, 
    332, 362, 351, 344, 352, 353, 326, 344
};

__device__ int egBishopTable[64] = {
    283, 276, 286, 289, 290, 288, 280, 273, 
    289, 293, 304, 285, 294, 284, 293, 283, 
    299, 289, 297, 296, 295, 303, 297, 301, 
    294, 306, 309, 306, 311, 307, 300, 299, 
    291, 300, 310, 316, 304, 307, 294, 288, 
    285, 294, 305, 307, 310, 300, 290, 282, 
    283, 279, 290, 296, 301, 288, 282, 270, 
    274, 288, 274, 292, 288, 281, 292, 280
};

__device__ int mgRookTable[64] = {
    509, 519, 509, 528, 540, 486, 508, 520, 
    504, 509, 535, 539, 557, 544, 503, 521, 
    472, 496, 503, 513, 494, 522, 538, 493, 
    453, 466, 484, 503, 501, 512, 469, 457, 
    441, 451, 465, 476, 486, 470, 483, 454, 
    432, 452, 461, 460, 480, 477, 472, 444, 
    433, 461, 457, 468, 476, 488, 471, 406, 
    458, 464, 478, 494, 493, 484, 440, 451
};

__device__ int egRookTable[64] = {
    525, 522, 530, 527, 524, 524, 520, 517, 
    523, 525, 525, 523, 509, 515, 520, 515, 
    519, 519, 519, 517, 516, 509, 507, 509, 
    516, 515, 525, 513, 514, 513, 511, 514, 
    515, 517, 520, 516, 507, 506, 504, 501, 
    508, 512, 507, 511, 505, 500, 504, 496, 
    506, 506, 512, 514, 503, 503, 501, 509, 
    503, 514, 515, 511, 507, 499, 516, 492
};

__device__ int mgQueenTable[64] = {
     997, 1025, 1054, 1037, 1084, 1069, 1068, 1070, 
    1001,  986, 1020, 1026, 1009, 1082, 1053, 1079, 
    1012, 1008, 1032, 1033, 1054, 1081, 1072, 1082,
     998,  998, 1009, 1009, 1024, 1042, 1023, 1026, 
    1016,  999, 1016, 1015, 1023, 1021, 1028, 1022, 
    1011, 1027, 1014, 1023, 1020, 1027, 1039, 1030, 
     990, 1017, 1036, 1027, 1033, 1040, 1022, 1026, 
    1024, 1007, 1016, 1035, 1010, 1000,  994,  975
};

__device__ int egQueenTable[64] = {
    927, 958, 958, 963, 963, 955, 946, 956, 
    919, 956, 968, 977, 994, 961, 966, 936, 
    916, 942, 945, 985, 983, 971, 955, 945, 
    939, 958, 960, 981, 993, 976, 993, 972, 
    918, 964, 955, 983, 967, 970, 975, 959, 
    920, 909, 951, 942, 945, 953, 946, 941, 
    914, 913, 906, 920, 920, 913, 900, 904, 
    903, 908, 914, 893, 931, 904, 916, 895
};

__device__ int mgKingTable[64] = {
    2935, 3023, 3016, 2985, 2944, 2966, 3002, 3013, 
    3029, 2999, 2980, 2993, 2992, 2996, 2962, 2971, 
    2991, 3024, 3002, 2984, 2980, 3006, 3022, 2978, 
    2983, 2980, 2988, 2973, 2970, 2975, 2986, 2964, 
    2951, 2999, 2973, 2961, 2954, 2956, 2967, 2949, 
    2986, 2986, 2978, 2954, 2956, 2970, 2985, 2973, 
    3001, 3007, 2992, 2936, 2957, 2984, 3009, 3008, 
    2985, 3036, 3012, 2946, 3008, 2972, 3024, 3014
};

__device__ int egKingTable[64] = {
    2926, 2965, 2982, 2982, 2989, 3015, 3004, 2983, 
    2988, 3017, 3014, 3017, 3017, 3038, 3023, 3011, 
    3010, 3017, 3023, 3015, 3020, 3045, 3044, 3013, 
    2992, 3022, 3024, 3027, 3026, 3033, 3026, 3003, 
    2982, 2996, 3021, 3024, 3027, 3023, 3009, 2989, 
    2981, 2997, 3011, 3021, 3023, 3016, 3007, 2991, 
    2973, 2989, 3004, 3013, 3014, 3004, 2995, 2983, 
    2947, 2966, 2979, 2989, 2972, 2986, 2976, 2957
};

__device__ int evaluatePosition(
    const pos64& whitePawns, const pos64& whiteBishops,
    const pos64& whiteKnights, const pos64& whiteRooks,
    const pos64& whiteQueens, const pos64& whiteKings, const pos64& blackPawns,
    const pos64& blackBishops, const pos64& blackKnights,
    const pos64& blackRooks, const pos64& blackQueens,
    const pos64& blackKings) {
    int gamePhase = 0;
    int midgameScore = 0;
    int endgameScore = 0;

    /* evaluate each piece */
    pos64 pos = 1;
    for (int i = 0; i < 64; i++) {
        int iNegated = i ^ 56;
        if ((whitePawns & pos) != 0) {
            midgameScore += mgPawnTable[iNegated];
            endgameScore += egPawnTable[iNegated];
        }
        if ((blackPawns & pos) != 0) {
            midgameScore -= mgPawnTable[i];
            endgameScore -= egPawnTable[i];
        }
        if ((whiteBishops & pos) != 0) {
            midgameScore += mgBishopTable[iNegated];
            endgameScore += egBishopTable[iNegated];
            gamePhase += 1;
        }
        if ((blackBishops & pos) != 0) {
            midgameScore -= mgBishopTable[i];
            endgameScore -= egBishopTable[i];
            gamePhase += 1;
        }
        if ((whiteKnights & pos) != 0) {
            midgameScore += mgKnightTable[iNegated];
            endgameScore += egKnightTable[iNegated];
            gamePhase += 1;
        }
        if ((blackKnights & pos) != 0) {
            midgameScore -= mgKnightTable[i];
            endgameScore -= egKnightTable[i];
            gamePhase += 1;
        }
        if ((whiteRooks & pos) != 0) {
            midgameScore += mgRookTable[iNegated];
            endgameScore += egRookTable[iNegated];
            gamePhase += 2;
        }
        if ((blackRooks & pos) != 0) {
            midgameScore -= mgRookTable[i];
            endgameScore -= egRookTable[i];
            gamePhase += 2;
        }
        if ((whiteQueens & pos) != 0) {
            midgameScore += mgQueenTable[iNegated];
            endgameScore += egQueenTable[iNegated];
            gamePhase += 4;
        }
        if ((blackQueens & pos) != 0) {
            midgameScore -= mgQueenTable[i];
            endgameScore -= egQueenTable[i];
            gamePhase += 4;
        }
        if ((whiteKings & pos) != 0) {
            midgameScore += mgKingTable[iNegated];
            endgameScore += egKingTable[iNegated];
        }
        if ((blackKings & pos) != 0) {
            midgameScore -= mgKingTable[i];
            endgameScore -= egKingTable[i];
        }
        pos <<= 1;
    }

    if (gamePhase > 24) gamePhase = 24;
    return (midgameScore * gamePhase + endgameScore * (24 - gamePhase)) / 24.0;
}
}  // namespace EVALUATION