#ifndef ERROR_H_INCLUDED
#define ERROR_H_INCLUDED

#include <stdlib.h>

#define ERR(source) (perror(source),\
            fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
            exit(EXIT_FAILURE))

#endif