#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef float ttype[3];

//OR GATE 
float or_train [4][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};


//AND GATE:
float and_train [4][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1}
};


//NAND GATE:
float nand_train [4][3] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};

float (*train2[3])[4][3] = {&or_train, &and_train, &nand_train};

#define TRAIN_COUNT 4 // (int)(sizeof(train)/sizeof(train[0]))


//Activation functions.
float sigmoidf(float x){

    return 1.f/ (1.f + expf(-x));
}


float cost_function(float w1, float w2, float bias){
    float result = 0.0f;
    float (*train)[4][3] = train2[1];

    for(size_t i = 0 ;i < TRAIN_COUNT; ++i ){

        float x1 = (*train)[i][0];
        float x2 = (*train)[i][1];

        // fit our value in our model.
        float y = sigmoidf(x1*w1 + x2*w2 + bias); //x1*w1 + x2*w2; 

        float diff = y - (*train)[i][2];
        result += diff*diff;

        // printf("actual: %f, expcted: %f\n", y, train[i][1]);

    }
    result /= TRAIN_COUNT;
    return result;
}


float gen_rand_float(){
    return (float)rand()/(float) RAND_MAX;
}


int main(){

    //Test sigmoid function
    // for(float m = -10.f; m <= 10.f; m+=1.f){

    //     printf("m: %f =>>> sig(): %f\n\r", m, sigmoidf(m));

    // }
        float (*train)[4][3] = train2[1];

        for(size_t i = 0 ;i < TRAIN_COUNT; ++i ){

            float x1 = (*train)[i][0];
            float x2 = (*train)[i][1];

            // printf("actual: %f, expcted: %f\n", y, train[i][1]);
            printf("%f %f %f\n", x1, x2, (*train)[i][2] );
        }

        printf("..........Inner........\n");
    
    for(size_t i = 0 ;i < 3; i++ ){
        for(size_t j = 0 ;j < 4; j++ ){
                // printf("%f %f %f\n", (*train2[i])[j][0], (*train2[i])[j][1], (*train2[i])[j][2] );
        }
        // printf("..................\n");
    }

    srand(time(0));
    float w1 = gen_rand_float();
    float w2 = gen_rand_float();

    float b = gen_rand_float();

    float epslum = 1e-1;
    float learning_rate = 1e-1;


    for(int i = 0; i < 1000 * 1000; ++i){

        float c = cost_function(w1, w2, b);
        // printf("Weight1: %f, Weight2: %f, bias: %f, Cost Function: %f\n", w1, w2, b, cost_function(w1, w2, b));

        // for graph plotting 
        // printf("%f\n", c);

        float dw1 = ((cost_function(w1 + epslum, w2, b) - c) / epslum); //derivative equation: dw = weight_derivative_cost
        float dw2 = ((cost_function(w1, w2 + epslum, b) - c) / epslum); //derivative equation: dw = weight_derivative_cost
        float db =  ((cost_function(w1 , w2 , b + epslum) - c) / epslum); //derivative equation: db = bias_derivative_cost

        w1 -= learning_rate * dw1;
        w2 -= learning_rate * dw2;
        b -= learning_rate * db;

    }
    // printf("--------------------------------------------\n");
    // printf("Weight1: %f, Weight2: %f, bias: %f, Cost Function: %f\n", w1, w2, b, cost_function(w1, w2, b));


    //Test model performance
    for(size_t i = 0; i < 2; i++){
        for(size_t k = 0; k < 2; k++){
            printf("%zu: | %zu: = %f\n", i, k, sigmoidf(i*w1 + k*w2 + b)); 
        }
    }

    return 1;
}
