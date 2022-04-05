#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <cmath>
#include <stdint.h>
#include <functional>

#define ANALYSIS_MODE

#define ANALYSIS_RATE 100

#define ERR_INVALID_INPUT_STREAM_SIZE   "invalid input stream (%s): %llu is not a multiple of %llu\n"
#define ERR_INVALID_INPUT_STREAM_TYPE   "invalid input stream: input stream must be of type double\n"
#define ERR_UNMATCHED_ACT               "ERROR: unmatched activation function\n"
#define ERR_INPUT_STREAM                "ERROR: something went wrong when reading from %s\n"

#define DEFAULT_LEARNING_RATE   1e-2
#define LEARNING_RATE_DECAY     0.1

#define RELU(x) (((x) > 0.0) ? (x) : 0.001*(x))
#define D_RELU(x) (((x) > 0.0) ? 1.0 : 0.001)
#define SIGM_POS(x) (1.0 / (1.0 + exp(-x)))     // avoids overflow for positive input
#define SIGM_NEG(x) (exp(x) / (1.0 + exp(x)))   // avoids overflow for negative input

#define HE(in) sqrt(double (6.0 / (in)))
#define XAVIER(in, out) sqrt(double (6.0 / ((in) + (out))))

#define EPSILON 0.01

/*
* value between 0 and 1, the higher this value, the slower gradient descent reacts to change in the gradient
* the idea is to punch through oscillations by combining the gradient with recent gradients
*/
#define MOMENTUM_SCALE 0.7
#define UPDATE_MOMENTUM(s, m, g) m = s*m + (1.0 - s)*(g);

#define SIGN(T, val) ((T(0) < (val)) - ((val) < T(0)))  // pos -> 1, neg -> -1, 0 -> 0
#define DIFF_SIGN(x, y) (((x) < 0)^((y) < 0))
#define ABS(T, x) ((1 - 2*(T(0) > (x))) * (x))

#define OFFSET(T, ptr, off) &(((T*) ptr)[off])

#define ACT_ID      0
#define ACT_RELU    1
#define ACT_SIGM    2
#define ACT_TANH    3

#define OPT_MOMENTUM        0x01
/*
* update the params after a run over the whole dataset or after a run over one datapoint
*/
#define OPT_DELAYED_UPDATE  0x02

#define FLAG_ALLOC_TARGET   0x01
#define FLAG_OUTPUT_NAN     0x02

typedef struct {
    size_t layer_count;
    size_t* layer_size;
    double** layers;
    double** weights;
    double** constants;
    uint8_t* output_act;
    uint8_t inner_act;
} FC_DNN;

typedef struct {
    double** weights;
    double** constants;
} Gradient;

template <typename T>
struct InputData {
    size_t count;
    size_t size;
    T* data;
};

template <typename T>
struct TrainData {
    size_t count;
    size_t size;
    double scale;
    InputData<T>** data;
};

typedef struct {
    double initial;
    double** weights;
    double** constants;
} LearningRate;

FC_DNN* create_FC_DNN(size_t layer_count, size_t* layer_size);

FC_DNN* read_FC_DNN(std::ifstream* input);

void write_FC_DNN(FC_DNN* dnn, std::ofstream* output);

void free_FC_DNN(FC_DNN* dnn);

template <typename T>
InputData<T>* create_input_data() {
    InputData<T>* input_data = (InputData<T>*) malloc(sizeof(InputData<T>));
    input_data->size = 1;
    input_data->count = 0;
    input_data->data = (T*) malloc(input_data->size * sizeof(T));
    return input_data;
}

template <typename T>
void add_input_data(InputData<T>* input_data, std::ifstream* stream) {
    while (stream->good()) {
        if (input_data->count == input_data->size) {
            input_data->size *= 2;
            input_data->data = (T*) realloc(input_data->data, input_data->size * sizeof(T));
        }
        stream->read((char*) &(input_data->data[input_data->count]), sizeof(T));
        input_data->count++;
    }
    if (!stream->eof()) {
        fprintf(stderr, ERR_INPUT_STREAM, "training data");
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void free_input_data(InputData<T>* input_data) {
    free(input_data->data);
    free(input_data);
}

template <typename T>
TrainData<T>* create_train_data() {
    TrainData<T>* train_data = (TrainData<T>*) malloc(sizeof(TrainData<T>));
    train_data->size = 1;
    train_data->count = 0;
    train_data->scale = 1.0;
    train_data->data = (InputData<T>**) malloc(train_data->size * sizeof(InputData<T>*));
    return train_data;
}

template <typename T>
void add_train_data(TrainData<T>* train_data, std::ifstream* stream) {
    if (train_data->count == train_data->size) {
        train_data->size *= 2;
        train_data->data = (InputData<T>**) realloc(train_data->data, train_data->size * sizeof(InputData<T>*));
    }
    train_data->data[train_data->count] = create_input_data<T>();
    add_input_data(train_data->data[train_data->count], stream);
    train_data->count++;
}

template <typename T>
void free_train_data(TrainData<T>* train_data) {
    for (size_t i=0; i<train_data->count; i++) {
        free_input_data(train_data->data[i]);
    }
    free(train_data->data);
    free(train_data);
}

/*
* Initialize the parameters of the DNN
*/
void init_FC_DNN(FC_DNN* dnn);

/*
* calculate the output of the DNN
*/
double* consult_FC_DNN(FC_DNN* dnn);

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
);

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
);
#endif //NEURAL_NETWORK_H
