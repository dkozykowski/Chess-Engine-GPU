#ifndef MACROS_H_INCLUDED
#define MARCOS_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>

// Error handling
#define ERR(source) (perror(source),\
            fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
            exit(EXIT_FAILURE))

#define TOTAL_THREAD_NUM (int)1e6
#define START_POS "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"

#endif