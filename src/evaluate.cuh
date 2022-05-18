#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

// implementation of PeSTO's evaluation function 
// source: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

int  evaluate_position(char* position, int side2move);
void init_tables();

#endif