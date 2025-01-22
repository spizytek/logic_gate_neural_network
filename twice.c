#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float train [][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

#define TRAIN_COUNT (int)(sizeof(train)/sizeof(train[0]))

float gen_rand_float(){

    return (float)rand()/(float) RAND_MAX;
}

/*********************************************************************** *
 *              READ UP ABOUT DERIVATIVE OF A FUNCTION
***************************************************************************/
//We use this function to determine the overall performance of our model.

/*
*The derivative of a function tells us the information about the direction of which the function grows and the rate of change of the function
* at any given point.
Example:
- For a fucntion f(x) = x2
- f!(x) = 2x
- When x > 0, f!(x) = 2x > 0, so the function is increasing (moving up as x increases).
- When x < 0, f!(x) = 2x < 0, so the function is decreasing (moving down as x decreases).

- Input: x (param)
- Output: y 
- Weight: w
- Bias: b

x1, x2, x3 ....xn
w1, w2, w3 ....wn

y = x1w1 + x2w2 + x3w3 .... xnwn + b
*/


float cost_function(float param, float bias){
    float result = 0.0f;

    for(int i = 0 ;i < TRAIN_COUNT; ++i ){

        float x = train[i][0];
        // fit our value in our model.
        float y = x*param + bias; 
        float diff = y - train[i][1];
        result += diff*diff;

        // printf("actual: %f, expcted: %f\n", y, train[i][1]);
    }
    result /= TRAIN_COUNT;
    return result;
}

int main(void){

// We are using one parameter, w.
// y = am (model)

    // srand(time(NULL));
    srand(69);
    float param = gen_rand_float()* 10.0f;
    float b = gen_rand_float()*  5.0f;

    float epslum = 1e-3;
    float learning_rate = 1e-3;


   
    printf("%f\n",  cost_function(param, b));
    for(int i = 0; i < 1000; i++){
        float c = cost_function(param, b);

        float dw = ((cost_function(param + epslum, b) - c) / epslum); //derivative equation: dw = weight_derivative_cost
        float db = ((cost_function(param , b + epslum) - c) / epslum); //derivative equation: db = bias_derivative_cost

        param -= learning_rate*dw;
        b -= learning_rate*db;

        printf("Cost Function: %f, Param: %f\n", cost_function(param, b), param);
    }
    printf(".......................................\n");


    printf("%f\n", param);

}