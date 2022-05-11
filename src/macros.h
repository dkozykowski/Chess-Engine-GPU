#ifndef MACROS_H_INCLUDED
#define MARCOS_H_INCLUDED

#include <stdlib.h>

// Error handling
#define ERR(source) (perror(source),\
            fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
            exit(EXIT_FAILURE))

#define DEFAULT_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define MAX_FEN_LEN 90

#endif