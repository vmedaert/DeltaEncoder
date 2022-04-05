#include <cmath>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <functional>

#include "NeuralNetwork.h"

/*
* swap values between two variables, x and y are the pointers to the variables
*/
template <typename T> void swap(T* x, T* y) {
    T buff;
    memcpy(&buff, y, sizeof(T));
    memcpy(y, x, sizeof(T));
    memcpy(x, &buff, sizeof(T));
}

void divmod(int x, int y, int& q, int& r) {
    if (y == 0) {
        std::cerr << "division by zero\n";
        exit(EXIT_FAILURE);
    }
    q = (int) floor(double (x / y));
    r = x - q*y;
}

double normalized_random_double() {
    return 2.0 * ((double) rand() / RAND_MAX) - 1.0;
}

FC_DNN* create_FC_DNN(size_t layer_count, size_t* layer_size) {
    FC_DNN* dnn = (FC_DNN*) malloc(sizeof(FC_DNN));
    dnn->layer_count = layer_count;
    dnn->layer_size = (size_t*) malloc(layer_count * sizeof(size_t));
    memcpy(dnn->layer_size, layer_size, layer_count * sizeof(size_t));
    dnn->layers = (double**) malloc(layer_count * sizeof(double*));
    dnn->weights = (double**) malloc((layer_count - 1) * sizeof(double*));
    dnn->constants = (double**) malloc((layer_count - 1) * sizeof(double*));
    dnn->output_act = (uint8_t*) malloc(layer_size[layer_count - 1] * sizeof(uint8_t));
    for (size_t i=0; i<layer_count; i++) {
        dnn->layers[i] = (double*) malloc(layer_size[i] * sizeof(double));
    }
    for (size_t i=1; i<layer_count; i++) {
        dnn->constants[i-1] = (double*) malloc(layer_size[i] * sizeof(double));
    }
    for (size_t i=1; i<layer_count; i++) {
        dnn->weights[i-1] = (double*) malloc(layer_size[i] * layer_size[i-1] * sizeof(double));
    }
    return dnn;
}

FC_DNN* read_FC_DNN(std::ifstream* input) {
    FC_DNN* dnn = (FC_DNN*) malloc(sizeof(FC_DNN));
    input->read((char*) &(dnn->layer_count), sizeof(size_t));
    input->read((char*) &(dnn->inner_act), sizeof(uint8_t));
    dnn->layer_size = (size_t*) malloc(dnn->layer_count * sizeof(size_t));
    input->read((char*) dnn->layer_size, dnn->layer_count * sizeof(size_t));
    dnn->layers = (double**) malloc(dnn->layer_count * sizeof(double*));
    dnn->weights = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    dnn->constants = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    dnn->output_act = (uint8_t*) malloc(dnn->layer_size[dnn->layer_count - 1] * sizeof(uint8_t));
    input->read((char*) dnn->output_act, dnn->layer_size[dnn->layer_count - 1] * sizeof(uint8_t));
    for (size_t i=0; i<dnn->layer_count; i++) {
        #ifdef ANALYSIS_MODE
        printf("layer %lld size = %lld\n", i, dnn->layer_size[i]);
        fflush(stdout);
        #endif
        dnn->layers[i] = (double*) malloc(dnn->layer_size[i] * sizeof(double));
    }
    for (size_t i=1; i<dnn->layer_count; i++) {
        dnn->constants[i-1] = (double*) malloc(dnn->layer_size[i] * sizeof(double));
        input->read((char*) dnn->constants[i-1], dnn->layer_size[i] * sizeof(double));
    }
    for (size_t i=1; i<dnn->layer_count; i++) {
        dnn->weights[i-1] = (double*) malloc(dnn->layer_size[i] * dnn->layer_size[i-1] * sizeof(double));
        input->read((char*) dnn->weights[i-1], dnn->layer_size[i] * dnn->layer_size[i-1] * sizeof(double));
    }
    if (!input->good()) {
        fprintf(stderr, ERR_INPUT_STREAM, "neural network file");
        exit(EXIT_FAILURE);
    }
    return dnn;
}

void write_FC_DNN(FC_DNN* dnn, std::ofstream* output) {
    output->write((char*) &(dnn->layer_count), sizeof(size_t));
    output->write((char*) &(dnn->inner_act), sizeof(uint8_t));
    output->write((char*) dnn->layer_size, dnn->layer_count * sizeof(size_t));
    output->write((char*) dnn->output_act, dnn->layer_size[dnn->layer_count - 1] * sizeof(uint8_t));
    for (size_t i=1; i<dnn->layer_count; i++) {
        output->write((char*) dnn->constants[i-1], dnn->layer_size[i] * sizeof(double));
    }
    for (size_t i=1; i<dnn->layer_count; i++) {
        output->write((char*) dnn->weights[i-1], dnn->layer_size[i] * dnn->layer_size[i-1] * sizeof(double));
    }
}

void free_FC_DNN(FC_DNN* dnn) {
    for (size_t i=0; i<dnn->layer_count; i++) {
        free(dnn->layers[i]);
    }
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(dnn->constants[i]);
    }
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(dnn->weights[i]);
    }
    free(dnn->layer_size);
    free(dnn->layers);
    free(dnn->weights);
    free(dnn->constants);
    free(dnn->output_act);
    free(dnn);
}

double** create_update_buff(FC_DNN* dnn) {
    double** update_buff = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        update_buff[i] = (double*) malloc(dnn->layer_size[dnn->layer_count - 1 - i] * sizeof(double));
    }
    return update_buff;
}

void free_update_buff(FC_DNN* dnn, double** update_buff) {
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(update_buff[i]);
    }
    free(update_buff);
}

void create_gradient(FC_DNN* dnn, Gradient& grad) {
    grad.weights = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    for (size_t i=1; i<dnn->layer_count; i++) {
        grad.weights[i-1] = (double*) malloc(dnn->layer_size[i] * dnn->layer_size[i-1] * sizeof(double));
    }
    grad.constants = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    for (size_t i=1; i<dnn->layer_count; i++) {
        grad.constants[i-1] = (double*) malloc(dnn->layer_size[i] * sizeof(double));
    }
}

void init_gradient(FC_DNN* dnn, Gradient grad) {
    for (size_t i=1; i<dnn->layer_count; i++) {
        memset(grad.weights[i-1], 0.0, dnn->layer_size[i] * dnn->layer_size[i-1] * sizeof(double));
    }
    for (size_t i=1; i<dnn->layer_count; i++) {
        memset(grad.constants[i-1], 0.0, dnn->layer_size[i] * sizeof(double));
    }
}

void free_gradient(FC_DNN* dnn, Gradient grad) {
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(grad.weights[i]);
    }
    free(grad.weights);
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(grad.constants[i]);
    }
    free(grad.constants);
}

void create_learning_rate(FC_DNN* dnn, LearningRate& learning_rate) {
    learning_rate.initial = DEFAULT_LEARNING_RATE;
    learning_rate.weights = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    for (size_t i=1; i<dnn->layer_count; i++) {
        learning_rate.weights[i-1] = (double*) malloc(dnn->layer_size[i] * dnn->layer_size[i-1] * sizeof(double));
    }
    learning_rate.constants = (double**) malloc((dnn->layer_count - 1) * sizeof(double*));
    for (size_t i=1; i<dnn->layer_count; i++) {
        learning_rate.constants[i-1] = (double*) malloc(dnn->layer_size[i] * sizeof(double));
    }
}

void init_learning_rate(FC_DNN* dnn, LearningRate learning_rate) {
    for (size_t i=1; i<dnn->layer_count; i++) {
        for (size_t j=0; j < dnn->layer_size[i] * dnn->layer_size[i-1]; j++) {
            learning_rate.weights[i-1][j] = learning_rate.initial;
        }
    }
    for (size_t i=1; i<dnn->layer_count; i++) {
        for (size_t j=0; j < dnn->layer_size[i]; j++) {
            learning_rate.constants[i-1][j] = learning_rate.initial;
        }
    }
}

void free_learning_rate(FC_DNN* dnn, LearningRate learning_rate) {
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(learning_rate.weights[i]);
    }
    free(learning_rate.weights);
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        free(learning_rate.constants[i]);
    }
    free(learning_rate.constants);
}

double init_param_cof(FC_DNN* dnn, size_t layer, uint8_t act) {
    switch(act) {
        case ACT_ID:
            return HE(dnn->layer_size[layer-1]);
        case ACT_RELU:
            return HE(dnn->layer_size[layer-1]);
        case ACT_SIGM:
            return XAVIER(dnn->layer_size[layer-1], dnn->layer_size[layer]);
        case ACT_TANH:
            return XAVIER(dnn->layer_size[layer-1], dnn->layer_size[layer]);
        default:
            std::cerr << ERR_UNMATCHED_ACT;
            exit(EXIT_FAILURE);
    }
}

double act_deriv(double x, uint8_t act) {
    switch(act) {
        case ACT_ID:
            return 1;
        case ACT_RELU:
            return D_RELU(x);
        case ACT_SIGM:
            x = (x < 0) ? SIGM_NEG(x) : SIGM_POS(x);
            return x * (1 - x);
        case ACT_TANH:
            return 1 / pow(cosh(x), 2);
        default:
            std::cerr << ERR_UNMATCHED_ACT;
            exit(EXIT_FAILURE);
    }
}

/*
* Initialize the parameters of the DNN
*/
void init_FC_DNN(FC_DNN* dnn) {
    double temp;
    for (size_t i=1; i<dnn->layer_count; i++) {
        memset(dnn->constants[i-1], 0, dnn->layer_size[i] * sizeof(double));
    }
    for (size_t i=1; i < dnn->layer_count - 1; i++) {
        temp = init_param_cof(dnn, i, dnn->inner_act);
        for (size_t j=0; j<dnn->layer_size[i]; j++) {
            for (size_t k=0; k<dnn->layer_size[i-1]; k++) {
                dnn->weights[i-1][j*dnn->layer_size[i-1] + k] = normalized_random_double() * temp;
            }
        }
    }
    for (size_t i=0; i<dnn->layer_size[dnn->layer_count-1]; i++) {
        temp = init_param_cof(dnn, dnn->layer_count - 1, dnn->output_act[i]);
        for (size_t j=0; j<dnn->layer_size[dnn->layer_count-1]; j++) {
            for (size_t k=0; k<dnn->layer_size[dnn->layer_count-2]; k++) {
                dnn->weights[dnn->layer_count-2][j*dnn->layer_size[dnn->layer_count-2] + k] = normalized_random_double() * temp;
            }
        }
    }
}

/*
* calculate the output of the DNN
*/
double* consult_FC_DNN(FC_DNN* dnn) {
    double* input;
    double* output;
    double* weights;
    size_t input_size;
    size_t output_size;
    for (size_t i=1; i<dnn->layer_count; i++) {
        input = dnn->layers[i-1];
        output = dnn->layers[i];
        weights = dnn->weights[i-1];
        input_size = dnn->layer_size[i-1];
        output_size = dnn->layer_size[i];
        for (size_t j=0; j<output_size; j++) {
            output[j] = dnn->constants[i-1][j];
            for (size_t k=0; k<input_size; k++) {
                output[j] += weights[j*input_size + k] * input[k];
            }
            switch((i == dnn->layer_count - 1) ? dnn->output_act[j] : dnn->inner_act) {
                case ACT_ID:
                    break;
                case ACT_RELU:
                    output[j] = RELU(output[j]);
                    break;
                case ACT_SIGM:
                    output[j] = (output[j] < 0) ? SIGM_NEG(output[j]) : SIGM_POS(output[j]);
                    break;
                case ACT_TANH:
                    output[j] = tanh(output[j]);
                    break;
                default:
                    std::cerr << ERR_UNMATCHED_ACT;
                    exit(EXIT_FAILURE);
            }
        }
    }
    return dnn->layers[dnn->layer_count - 1];
}

/*
* update the gradient
*/
void update_FC_DNN(
    FC_DNN* dnn,
    const std::function<double(double*, double*, size_t, size_t)>& err_func_deriv,
    double* target,
    Gradient* grad,
    Gradient* momentum,
    LearningRate* learning_rate,
    double** update_buff,
    uint8_t opt
) {
    double temp;
    // calculate the intermediate results that are recursively reused
    // output layer
    for (size_t j=0; j<dnn->layer_size[dnn->layer_count - 1]; j++) {
        // calculate derived activation function argument
        update_buff[0][j] = dnn->constants[dnn->layer_count - 2][j];
        for (size_t k=0; k<dnn->layer_size[dnn->layer_count - 2]; k++) {
            update_buff[0][j] += dnn->weights[dnn->layer_count - 2][j*dnn->layer_size[dnn->layer_count - 2] + k] * dnn->layers[dnn->layer_count - 2][k];
        }
        // apply derived activation function
        update_buff[0][j] = act_deriv(update_buff[0][j], dnn->output_act[j]);
        // calculate and store intermediate result
        update_buff[0][j] *= err_func_deriv(dnn->layers[dnn->layer_count - 1], target, dnn->layer_size[dnn->layer_count - 1], j);
    }
    // hidden layers
    for (size_t i = dnn->layer_count - 2; i>0; i--) {
        for (size_t j=0; j<dnn->layer_size[i]; j++) {
            // calculate derived activation function argument
            update_buff[dnn->layer_count - 1 - i][j] = dnn->constants[i-1][j];
            for (size_t k=0; k<dnn->layer_size[i-1]; k++) {
                update_buff[dnn->layer_count - 1 - i][j] += dnn->weights[i-1][j*dnn->layer_size[i-1] + k] * dnn->layers[i-1][k];
            }
            // apply derived activation function
            update_buff[dnn->layer_count - 1 - i][j] = act_deriv(update_buff[dnn->layer_count - 1 - i][j], dnn->inner_act);
            // reuse intermediate results of previous layer (actually next layer but previously calculated values)
            temp = 0;
            for (size_t k=0; k<dnn->layer_size[i+1]; k++) {
                temp += update_buff[dnn->layer_count - 2 - i][k] * dnn->weights[i][k*dnn->layer_size[i] + j];
            }
            // calculate and store intermediate result
            update_buff[dnn->layer_count - 1 - i][j] *= temp;
        }
    }
    // update the gradient using the already calculated intermediate results
    for (size_t i = dnn->layer_count - 1; i>0; i--) {
        for (size_t j=0; j<dnn->layer_size[i]; j++) {
            for (size_t k=0; k<dnn->layer_size[i-1]; k++) {
                if (opt & OPT_DELAYED_UPDATE) {
                    grad->weights[i-1][j*dnn->layer_size[i-1] + k] += update_buff[dnn->layer_count - 1 - i][j] * dnn->layers[i-1][k];
                } else {
                    temp = update_buff[dnn->layer_count - 1 - i][j] * dnn->layers[i-1][k];
                    grad->weights[i-1][j*dnn->layer_size[i-1] + k] += temp;
                    if (momentum) {
                        UPDATE_MOMENTUM(MOMENTUM_SCALE, momentum->weights[i-1][j*dnn->layer_size[i-1] + k], temp)
                        dnn->weights[i-1][j*dnn->layer_size[i-1] + k] -= learning_rate->weights[i-1][j*dnn->layer_size[i-1] + k] * momentum->weights[i-1][j*dnn->layer_size[i-1] + k];
                    } else {
                        dnn->weights[i-1][j*dnn->layer_size[i-1] + k] -= learning_rate->weights[i-1][j*dnn->layer_size[i-1] + k] * temp;
                    }
                }
            }
            grad->constants[i-1][j] += update_buff[dnn->layer_count - 1 - i][j];
            if (!(opt & OPT_DELAYED_UPDATE)) {
                if (momentum) {
                    UPDATE_MOMENTUM(MOMENTUM_SCALE, momentum->constants[i-1][j], update_buff[dnn->layer_count - 1 - i][j])
                    dnn->constants[i-1][j] -= learning_rate->constants[i-1][j] * momentum->constants[i-1][j];
                } else {
                    dnn->constants[i-1][j] -= learning_rate->constants[i-1][j] * update_buff[dnn->layer_count - 1 - i][j];
                }
            }
        }
    }
}

/*
* updates the params by applying the gradient or momentum
* (is only used with delayed updates)
*/
void apply_gradient(
    FC_DNN* dnn,
    LearningRate learning_rate,
    Gradient grad,
    Gradient total_grad,
    Gradient* momentum
) {
     for (size_t i=0; i < dnn->layer_count - 1; i++) {
        for (size_t j=0; j<dnn->layer_size[i+1]; j++) {
            for (size_t k=0; k<dnn->layer_size[i]; k++) {
                if (momentum) {
                    UPDATE_MOMENTUM(MOMENTUM_SCALE, momentum->weights[i][j*dnn->layer_size[i] + k], grad.weights[i][j*dnn->layer_size[i] + k])
                    dnn->weights[i][j*dnn->layer_size[i] + k] -= learning_rate.weights[i][j*dnn->layer_size[i] + k] * momentum->weights[i][j*dnn->layer_size[i] + k];
                } else {
                    dnn->weights[i][j*dnn->layer_size[i] + k] -= learning_rate.weights[i][j*dnn->layer_size[i] + k] * grad.weights[i][j*dnn->layer_size[i] + k];
                }
                total_grad.weights[i][j*dnn->layer_size[i] + k] += grad.weights[i][j*dnn->layer_size[i] + k];
                grad.weights[i][j*dnn->layer_size[i] + k] = 0;
            }
            if (momentum) {
                UPDATE_MOMENTUM(MOMENTUM_SCALE, momentum->constants[i][j], grad.constants[i][j])
                dnn->constants[i][j] -= learning_rate.constants[i][j] * momentum->constants[i][j];
            } else {
                dnn->constants[i][j] -= learning_rate.constants[i][j] * grad.constants[i][j];
            }
            total_grad.constants[i][j] += grad.constants[i][j];
            grad.constants[i][j] = 0;
        }
    }
}

/*
* trains the DNN on data
*/
void train_FC_DNN(
    FC_DNN* dnn,
    const std::function<double(double*, double*, size_t)>& err_func,
    const std::function<double(double*, double*, size_t, size_t)>& err_func_deriv,
    const std::function<bool(size_t*, double*&, uint8_t&)>& read_train_data,
    size_t max_iter_count,
    double err_bound,
    size_t update_rate,
    uint8_t opt
) {
    Gradient grad, total_grad, prev_total_grad, momentum;
    LearningRate learning_rate;
    double** update_buff = create_update_buff(dnn);
    double* target;
    double err;
    double prev_err;
    size_t run;
    size_t data_pointer[2];
    size_t iter_c;
    uint8_t flags = FLAG_OUTPUT_NAN;
    if (update_rate == 1) {
        opt &= (OPT_DELAYED_UPDATE^UINT_MAX);
    } else {
        opt |= (OPT_DELAYED_UPDATE);
    }
    create_learning_rate(dnn, learning_rate);
    create_gradient(dnn, total_grad);
    create_gradient(dnn, prev_total_grad);
    if (opt & OPT_DELAYED_UPDATE) create_gradient(dnn, grad);
    if (opt & OPT_MOMENTUM) create_gradient(dnn, momentum);
    while (flags & FLAG_OUTPUT_NAN) {
        flags = 0x00;
        err = 0;
        prev_err = 0;
        run = 0;
        init_FC_DNN(dnn);
        init_learning_rate(dnn, learning_rate);
        init_gradient(dnn, total_grad);
        init_gradient(dnn, prev_total_grad);
        if (opt & OPT_DELAYED_UPDATE) init_gradient(dnn, grad);
        if (opt & OPT_MOMENTUM) init_gradient(dnn, momentum);
        while (run < max_iter_count && (run < 2 || (ABS(double, err - prev_err) > err_bound))) {
            prev_err = err;
            err = 0;
            // update learning rates
            for (size_t i=0; i < dnn->layer_count - 1; i++) {
                for (size_t j=0; j<dnn->layer_size[i+1]; j++) {
                    for (size_t k=0; k<dnn->layer_size[i]; k++) {
                        if (DIFF_SIGN(total_grad.weights[i][j*dnn->layer_size[i] + k], prev_total_grad.weights[i][j*dnn->layer_size[i] + k])) {
                            learning_rate.weights[i][j*dnn->layer_size[i] + k] *= LEARNING_RATE_DECAY;
                        }
                    }
                    if (DIFF_SIGN(total_grad.constants[i][j], prev_total_grad.constants[i][j])) {
                        learning_rate.constants[i][j] *= LEARNING_RATE_DECAY;
                    }
                }
            }
            swap(&total_grad, &prev_total_grad);
            init_gradient(dnn, total_grad);
            memset(data_pointer, 0, 2*sizeof(size_t));
            iter_c = 0;
            while (read_train_data(data_pointer, target, flags)) {
                // calculate the output
                consult_FC_DNN(dnn);
                for (size_t i=0; i<dnn->layer_size[dnn->layer_count - 1]; i++) {
                    if (std::isnan(dnn->layers[dnn->layer_count - 1][i])) {
                        flags |= FLAG_OUTPUT_NAN;
                    }
                }
                // update error
                err += err_func(dnn->layers[dnn->layer_count - 1], target, dnn->layer_size[dnn->layer_count - 1]);
                // update gradient
                update_FC_DNN(
                    dnn,
                    err_func_deriv,
                    target,
                    (opt & OPT_DELAYED_UPDATE) ? &grad : &total_grad,
                    (opt & OPT_MOMENTUM) ? &momentum : NULL,
                    &learning_rate,
                    update_buff,
                    opt
                );
                if ((opt & OPT_DELAYED_UPDATE) && !(iter_c % update_rate)) {
                    apply_gradient(dnn, learning_rate, grad, total_grad, (opt & OPT_MOMENTUM) ? &momentum : NULL);
                }
                iter_c++;
            }
            if ((opt & OPT_DELAYED_UPDATE) && (iter_c - 1) % update_rate) {
                apply_gradient(dnn, learning_rate, grad, total_grad, (opt & OPT_MOMENTUM) ? &momentum : NULL);
            }
            err /= iter_c;
            #ifdef ANALYSIS_MODE
            if (!(run % ANALYSIS_RATE)) {
                printf("error: %f\n", err);
                fflush(stdout);
            }
            #endif
            run++;
        }
        if (flags & FLAG_OUTPUT_NAN) {
            learning_rate.initial *= LEARNING_RATE_DECAY;
            #ifdef ANALYSIS_MODE
            std::cout << "nans produced, lowering initial learning rate to " << learning_rate.initial << '\n';
            #endif
        }
    }
    // free memory
    free_update_buff(dnn, update_buff);
    free_learning_rate(dnn, learning_rate);
    free_gradient(dnn, total_grad);
    free_gradient(dnn, prev_total_grad);
    if (opt & OPT_DELAYED_UPDATE) free_gradient(dnn, grad);
    if (opt & OPT_MOMENTUM) free_gradient(dnn, momentum);
    if (flags & FLAG_ALLOC_TARGET) free(target);
}