#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;

} Xor;

const float epslum = 1e-1;
const float learning_rate = 1e-1;

//Activation functions.
float sigmoidf(float x){
    return 1.f/ (1.f + expf(-x));
}


float forward(Xor xg, float x1, float x2){
    float or_neuron = sigmoidf((xg.or_w1 * x1) + (xg.or_w2 * x2) + xg.or_b);
    float nand_neuron = sigmoidf((xg.nand_w1 * x1) + (xg.nand_w2 * x2) + xg.nand_b);

    return sigmoidf((xg.and_w1 * or_neuron) + (xg.and_w2 * nand_neuron) + xg.and_b);
}

typedef float ttype[3];

//OR GATE 
ttype xor_train [] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};
ttype *train  = xor_train;
size_t TRAIN_COUNT = 4;


float cost_function(Xor m){
    float result = 0.0f;

    for(size_t i = 0 ;i < TRAIN_COUNT; ++i ){
        float x1 = train[i][0];                                                                                                                                                                                      
        float x2 = train[i][1];
        // fit our value in our model.
        float y = forward(m, x1, x2);//x1*w1 + x2*w2; 

        float diff = y - train[i][2];
        result += diff*diff;
    }
    result /= TRAIN_COUNT;
    return result;
}

float gen_rand_float(){
    return (float)rand()/(float) RAND_MAX;
}

Xor populate_data(){
    Xor m;
    m.and_b = gen_rand_float();
    m.and_w1= gen_rand_float();
    m.and_w2 = gen_rand_float();
    m.nand_b = gen_rand_float();
    m.nand_w1 = gen_rand_float();
    m.nand_w2 = gen_rand_float();
    m.or_b  = gen_rand_float();
    m.or_w1 = gen_rand_float();
    m.or_w2 = gen_rand_float();

    return m;
}

void print_data(Xor mit){
    printf("and_b: %f, and_w1: %f, and_w2:  %f\n", mit.and_b, mit.and_w1, mit.and_w2);
    printf("nand_b:%f, nand_w1:%f, nand_w2: %f\n", mit.nand_b, mit.nand_w1, mit.nand_w2);
    printf("or_b:  %f, or_w1:  %f, or_w2:   %f\n", mit.or_b, mit.or_w1, mit.or_w2);
}


Xor finite_diff(Xor m){

    Xor g;
    float c = cost_function(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += epslum;
    g.or_w1 = ((cost_function(m) - c) / epslum);
    m.or_w1 = saved;


    saved = m.or_w2;
    m.or_w2 += epslum;
    g.or_w2 = ((cost_function(m) - c) / epslum);
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += epslum;
    g.or_b = ((cost_function(m) - c) / epslum);
    m.or_b = saved;

// For and
    saved = m.and_w1;
    m.and_w1 += epslum;
    g.and_w1 = ((cost_function(m) - c) / epslum);
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += epslum;
    g.and_w2 = ((cost_function(m) - c) / epslum);
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += epslum;
    g.and_b = ((cost_function(m) - c) / epslum);
    m.and_b = saved;

// For nand
    saved = m.nand_w1;
    m.nand_w1 += epslum;
    g.nand_w1 = ((cost_function(m) - c) / epslum);
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += epslum;
    g.nand_w2 = ((cost_function(m) - c) / epslum);
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += epslum;
    g.nand_b = ((cost_function(m) - c) / epslum);
    m.nand_b = saved;

    return g;
}

Xor learn(Xor m, Xor g, float learning_rate){

    m.or_w1 -= learning_rate * g.or_w1;
    m.or_w2-= learning_rate * g.or_w2;
    m.or_b -= learning_rate * g.or_b;

    m.nand_w1 -= learning_rate * g.nand_w1;
    m.nand_w2 -=  learning_rate * g.nand_w2;
    m.nand_b -= learning_rate * g.nand_b;

    m.and_w1 -= learning_rate * g.and_w1;
    m.and_w2 -=  learning_rate * g.and_w2;
    m.and_b -= learning_rate * g.and_b;

    return m;
}

int main(){

    Xor m = populate_data();

    for(int i = 0 ; i < 1000*10000; i++){
        Xor g = finite_diff(m);
        m = learn(m, g, learning_rate);
        // printf("Cost Function: %f\n", cost_function(m));
        
    }

    for (size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            printf("%zu %zu | %f\n", i, j, forward(m, i , j));
        }
    }
    
    return EXIT_SUCCESS;
}