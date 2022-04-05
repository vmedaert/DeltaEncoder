#include <fstream>
#include <iostream>
#include <time.h>
#include <functional>
#include <cmath>
#include <cstring>

#include "DeltaEncoder.h"
#include "NeuralNetwork.h"
#include "BitHandler.h"

static DeltaEncoderArgs args;

#ifdef DEBUG_MODE
static char bitstring_buff[sizeof(int64_t)*8 + 1];
#endif

const std::function<double(double*, double*, size_t)>
err_func = [](double* output, double* target, size_t size) {
    double err = 0;
    for (size_t i=0; i<size; i++) {
        err += pow(output[i] - target[i], 2);
    }
    return err;
};

const std::function<double(double*, double*, size_t, size_t)>
err_func_deriv = [](double* output, double* target, size_t size, size_t i) {
    return (output[i] - target[i]);
};

void train(std::ifstream* input, std::ofstream* output) {
    TrainData<int64_t>* train_data = create_train_data<int64_t>();
    size_t* layer_size;
    if (args.hidden_layer_size) {
        layer_size = (size_t*) malloc((args.hidden_layer_count + 2) * sizeof(size_t));
        memcpy((char*) &(layer_size[1]), (char*) args.hidden_layer_size, args.hidden_layer_count * sizeof(size_t));
        layer_size[args.hidden_layer_count + 1] = 1;
    } else {
        layer_size = (size_t*) malloc(4 * sizeof(size_t));
        layer_size[1] = HIDDEN_LAYER_1_SIZE;
        layer_size[2] = HIDDEN_LAYER_2_SIZE;
        layer_size[3] = 1;
    }
    layer_size[0] = args.input_layer_size;
    FC_DNN* dnn = create_FC_DNN((args.hidden_layer_size) ? args.hidden_layer_count + 2 : 4, layer_size);
    int64_t temp;
    dnn->inner_act = ACT_RELU;
    dnn->output_act[0] = ACT_ID;
    for (size_t i=0; i<args.input_c; i++) {
        std::cout << "loading " << args.input[i] << "\n";
        fflush(stdout);
        input->open(args.input[i], std::ios::binary);
        if (!input->is_open()) {
            std::cerr << "can't open " << args.input[i] << '\n';
            exit(EXIT_FAILURE);
        }
        add_train_data(train_data, input);
        input->close();
        input->clear();
    }
    // calculate scaling
    temp = 0;
    for (size_t i=0; i<train_data->count; i++) {
        for (size_t j=0; j<train_data->data[i]->count; j++) {
            if (ABS(int64_t, train_data->data[i]->data[j]) > temp) temp = ABS(int64_t, train_data->data[i]->data[j]);
        }
    }
    train_data->scale = 1.0 / temp;
    const std::function<bool(size_t*, double*&, uint8_t&)>
    read_train_data = [dnn, train_data](size_t* data_pointer, double*& target, uint8_t& flags) {
        bool output = data_pointer[1] < train_data->data[data_pointer[0]]->count - dnn->layer_size[0];
        int64_t* temp;
        // go to next non-empty set if current set has no data left
        while (!output && data_pointer[0] < train_data->count - 1) {
            data_pointer[0]++;
            data_pointer[1] = 0;
            output = train_data->data[data_pointer[0]]->count > dnn->layer_size[0];
        }
        if (output) {
            temp = &(train_data->data[data_pointer[0]]->data[data_pointer[1]]);
            // populate the input layer
            for (size_t i=0; i<dnn->layer_size[0]; i++) {
                dnn->layers[0][i] = train_data->scale * temp[i];
            }
            // update target (== pointer to expected output)
            if (!(flags & FLAG_ALLOC_TARGET)) {
                flags |= FLAG_ALLOC_TARGET;
                target = (double*) malloc(dnn->layer_size[dnn->layer_count - 1] * sizeof(double));
            }
            temp = &(temp[dnn->layer_size[0]]);
            for (size_t i=0; i<dnn->layer_size[dnn->layer_count - 1]; i++) {
                target[i] = train_data->scale * temp[i];
            }
            data_pointer[1]++;
        }
        return output;
    };
    train_FC_DNN(
        dnn,
        err_func,
        err_func_deriv,
        read_train_data,
        args.iteration_bound,
        args.err_bound,
        args.update_rate,
        OPT_MOMENTUM
    );
    // set the constants to the original scaling of the training data
    for (size_t i=0; i < dnn->layer_count - 1; i++) {
        for (size_t j=0; j<dnn->layer_size[i+1]; j++) {
            dnn->constants[i][j] /= train_data->scale;
        }
    }
    write_FC_DNN(dnn, output);
    // free memory
    free_train_data<int64_t>(train_data);
    free_FC_DNN(dnn);
    free(layer_size);
}

void encode(FC_DNN* dnn, std::ifstream* input, std::ofstream* output) {
    int64_t delta, temp;
    uint64_t utemp;
    size_t chunk_size = pow(2, 6 - args.len_bits);
    size_t counter;
    uint64_t mask = UINT64_MAX << (64 - chunk_size);
    BitWriter writer;
    #ifdef DEBUG_MODE
    bitstring_buff[sizeof(int64_t)*8] = '\0';
    #endif
    init_bit_writer(&writer, output);
    #ifdef USE_DELTA
    // populate input layer
    for (size_t i=0; i<dnn->layer_size[0]; i++) {
        input->read((char*) &temp, sizeof(int64_t));
        output->write((char*) &temp, sizeof(int64_t));
        dnn->layers[0][i] = (double) temp;
    }
    if (!input->good()) {
        if (input->eof()) {
            std::cerr << "ERR: file too small, need at least " << dnn->layer_size[0] * sizeof(int64_t) << " bytes\n";
        } else {
            fprintf(stderr, ERR_INPUT_STREAM, "input file");
        }
        exit(EXIT_FAILURE);
    }
    #endif
    input->read((char*) &temp, sizeof(int64_t));
    while (input->good()) {
        #ifdef USE_DELTA
        // predict value
        consult_FC_DNN(dnn);
        delta = floor(dnn->layers[dnn->layer_count - 1][0]) - temp;
        #ifdef DEBUG_MODE
        bitstring(bitstring_buff, (char*) &delta, sizeof(int64_t), big_endian());
        std::cout << bitstring_buff << '\n';
        std::cout << "delta = " << delta << " value = "
            << temp << " ratio = " << (double) ABS(double, (double) delta / temp) << '\n';
        #endif
        #else
        delta = temp,
        #endif
        // count the amount of chunks to be coded
        memcpy(&utemp, &delta, sizeof(uint64_t));
        counter = 0;
        if (utemp & MOST_SIG_BIT_64) utemp ^= UINT64_MAX;
        while (counter < (64 / chunk_size) - 1 && !(utemp & mask)) {
            utemp <<= chunk_size;
            counter++;
        }
        counter = (64 / chunk_size) - counter;
        // if most significant encoded bit equals sign bit
        if (!(((delta >> (counter*chunk_size - 1))^(delta >> 63)) & 0x1)) counter--;
        write_bits(&writer, (uint64_t*) &counter, args.len_bits);
        counter++;
        // mask out the bits that shouldn't be written
        if (delta & MOST_SIG_BIT_64) delta &= UINT64_MAX ^ (UINT64_MAX << counter * chunk_size);
        write_bits(&writer, (uint64_t*) &delta, counter * chunk_size);
        // shift input layer
        for (size_t i=0; i < dnn->layer_size[0] - 1; i++) {
            dnn->layers[0][i] = dnn->layers[0][i+1];
        }
        dnn->layers[0][dnn->layer_size[0] - 1] = (double) temp;
        input->read((char*) &temp, sizeof(int64_t));
    }
    flush_bits(&writer);
    if (!input->eof()) {
        fprintf(stderr, ERR_INPUT_STREAM, "input file");
        exit(EXIT_FAILURE);
    }
}

void decode(FC_DNN* dnn, std::ifstream* input, std::ofstream* output) {
    int64_t temp;
    size_t chunk_size = pow(2, 6 - args.len_bits);
    BitReader reader;
    uint8_t count;
    init_bit_reader(&reader, input);
    #ifdef DEBUG_MODE
    bitstring_buff[sizeof(int64_t)*8] = '\0';
    #endif
    #ifdef USE_DELTA
    // populate input layer
    for (size_t i=0; i<dnn->layer_size[0]; i++) {
        input->read((char*) &temp, sizeof(int64_t));
        output->write((char*) &temp, sizeof(int64_t));
        dnn->layers[0][i] = (double) temp;
    }
    if (!input->good()) {
        if (input->eof()) {
            std::cerr << "ERR: file too small, need at least " << dnn->layer_size[0] * sizeof(int64_t) << " bytes\n";
        } else {
            fprintf(stderr, ERR_INPUT_STREAM, "input file");
        }
        exit(EXIT_FAILURE);
    }
    #endif
    while (input->good()) {
        // read delta
        count = read_bits(&reader, args.len_bits) + 1;
        if (!input->good()) break;
        temp = (int64_t) read_bits(&reader, count * chunk_size);
        if (!input->good()) break;
        // restore sign
        temp <<= (64 - count*chunk_size);
        temp >>= (64 - count*chunk_size);
        #ifdef DEBUG_MODE
        bitstring(bitstring_buff, (char*) &temp, sizeof(int64_t), big_endian());
        std::cout << bitstring_buff << "\ndelta = " << temp;
        #endif
        #ifdef USE_DELTA
        // predict value
        consult_FC_DNN(dnn);
        // reconstruct original value
        temp = floor(dnn->layers[dnn->layer_count - 1][0]) - temp;
        output->write((char*) &temp, sizeof(int64_t));
        #ifdef DEBUG_MODE
        std::cout << " value = " << temp << '\n';
        #endif
        // shift input layer
        for (size_t i=0; i < dnn->layer_size[0] - 1; i++) {
            dnn->layers[0][i] = dnn->layers[0][i+1];
        }
        dnn->layers[0][dnn->layer_size[0] - 1] = (double) temp;
        #else
        output->write((char*) &temp, sizeof(int64_t));
        #endif
    }
    if (!input->eof()) {
        fprintf(stderr, ERR_INPUT_STREAM, "input file");
        exit(EXIT_FAILURE);
    }
}

void syntax_error(char* path) {
    char* name = path;
    size_t i = 0;
    while (path[i]) {
        if (path[i] == '/' || path[i] == '\\') name = &(path[i+1]);
        i++;
    }
    std::cerr << ERR_SYNTAX
        << name
        << " [options --"
        << CMD_OPT_INPUT_SIZE << "=INTEGER --" 
        << CMD_OPT_HIDDEN_SIZE << "=INTEGER* --"
        << CMD_OPT_ITER_BOUND << "=INTEGER --"
        << CMD_OPT_UPDATE_RATE << "=INTEGER --"
        << CMD_OPT_ERR_BOUND  << "=DOUBLE] "
        << CMD_TRAIN
        << " neural_network training_data*\n"
        << name << " [options --"
        << CMD_OPT_LEN_BITS << "=INTEGER[1..6]] ("
        << CMD_ENCODE << '|' << CMD_DECODE
        << ") neural_network in_file out_file\n";
    exit(EXIT_FAILURE);
}

void parse_args(int argc, char *argv[]) {
    char* key;
    char* val;
    char buff[PARSE_BUFF_SIZE];
    int i = 1;
    size_t j, k;
    size_t temp;
    args.iteration_bound = ITERATION_BOUND;
    args.update_rate = UPDATE_RATE;
    args.err_bound = ERR_BOUND;
    args.len_bits = LEN_BITS;
    args.input_layer_size = INPUT_SIZE;
    args.hidden_layer_count = 0;
    args.hidden_layer_size = NULL;
    while (i < argc && !strncmp(argv[i], "--", 2)) {
        j = 0;
        while (argv[i][j] != '=' && argv[i][j] != '\0') j++;
        if (argv[i][j]  == '=') {
            key = &(argv[i][2]);
            val = &(argv[i][j+1]);
            if (!strncmp(key, CMD_OPT_INPUT_SIZE, sizeof(CMD_OPT_INPUT_SIZE) - 1)) {
                args.input_layer_size = strtoul(val, NULL, 10);
            } else if (!strncmp(key, CMD_OPT_ITER_BOUND , sizeof(CMD_OPT_ITER_BOUND) - 1)) {
                args.iteration_bound = strtoul(val, NULL, 10);
            } else if (!strncmp(key, CMD_OPT_UPDATE_RATE, sizeof(CMD_OPT_UPDATE_RATE) - 1)) {
                args.update_rate = strtoul(val, NULL, 10);
            } else if (!strncmp(key, CMD_OPT_ERR_BOUND, sizeof(CMD_OPT_ERR_BOUND) - 1)) {
                args.err_bound = strtod(val, NULL);
            } else if (!strncmp(key, CMD_OPT_LEN_BITS, sizeof(CMD_OPT_LEN_BITS) - 1)) {
                args.len_bits = (uint8_t) strtoul(val, NULL, 10);
            } else if (!strncmp(key, CMD_OPT_HIDDEN_SIZE, sizeof(CMD_OPT_HIDDEN_SIZE) - 1)) {
                temp = 1;
                args.hidden_layer_size = (size_t*) malloc(sizeof(size_t));
                k = 0;
                while (k < PARSE_BUFF_SIZE - 1 && val[k] != '\0') {
                    while (k < PARSE_BUFF_SIZE - 1 && val[k] != ',' && val[k] != '\0') {
                        buff[k] = val[k];
                        k++;
                    }
                    buff[k] = '\0';
                    if (args.hidden_layer_count >= temp) {
                        temp *= 2;
                        args.hidden_layer_size = (size_t*) realloc(args.hidden_layer_size, temp * sizeof(size_t));
                    }
                    args.hidden_layer_size[args.hidden_layer_count] = strtoul(buff, NULL, 10);
                    args.hidden_layer_count++;
                    if (val[k] != '\0') {
                        val = &(val[k + 1]);
                        k = 0;
                    }
                }
            } else {
                std::cerr << "invalid option: " << key << '\n';
                syntax_error(argv[0]);
            }
        }
        i++;
    }
    if (!strncmp(argv[i], CMD_TRAIN, sizeof(CMD_TRAIN))) {
        args.command = ENUM_TRAIN;
    } else if (!strncmp(argv[i], CMD_ENCODE, sizeof(CMD_ENCODE))) {
        args.command = ENUM_ENCODE;
    } else if (!strncmp(argv[i], CMD_DECODE, sizeof(CMD_DECODE))) {
        args.command = ENUM_DECODE;
    } else {
        syntax_error(argv[0]);
    }
    i++;
    if (i >= argc) {
        fprintf(stderr, ERR_MISSING_FILE, "neural network");
        exit(EXIT_FAILURE);
    }
    args.dnn = argv[i];
    i++;
    if (i >= argc) {
        fprintf(stderr, ERR_MISSING_FILE, "input");
        exit(EXIT_FAILURE);
    }
    if (args.command == ENUM_TRAIN) {
        args.input_c = 0;
        args.input = (char**) malloc((argc - i) * sizeof(char*));
        while (i < argc) {
            args.input[args.input_c] = argv[i];
            args.input_c++;
            i++;
        }
    } else {
        args.input = (char**) malloc(sizeof(char*));
        args.input_c = 1;
        args.input[0] = argv[i];
    }
    if (args.command != ENUM_TRAIN) {
        i++;
        if (i >= argc) {
            fprintf(stderr, ERR_MISSING_FILE, "output");
            exit(EXIT_FAILURE);
        }
        args.output = argv[i];
    }
}

int main(int argc, char *argv[]) {
    FC_DNN* dnn;
    std::ifstream input;
    std::ofstream output;
    parse_args(argc, argv);
    srand(time(NULL));
    if (args.command == ENUM_TRAIN) {
        output.open(args.dnn, std::ios::binary);
        if (!output.is_open()) {
            std::cerr << "can't open " << args.dnn << '\n';
            exit(EXIT_FAILURE);
        }
        train(&input, &output);
        std::cout << "neural network saved in " << args.dnn << "\n";
        fflush(stdout);
        output.close();
    } else {
        std::cout << "loading " << args.dnn << "\n";
        fflush(stdout);
        input.open(args.dnn, std::ios::binary);
        if (!input.is_open()) {
            std::cerr << "can't open " << args.dnn << '\n';
            exit(EXIT_FAILURE);
        }
        dnn = read_FC_DNN(&input);
        input.close();
        input.clear(); // reset the error flags
        std::cout << "loading " << args.input[0] << "\n";
        fflush(stdout);
        input.open(args.input[0], std::ios::binary);
        if (!input.is_open()) {
            std::cerr << "can't open " << args.input[0] << '\n';
            exit(EXIT_FAILURE);
        }
        output.open(args.output, std::ios::binary);
        if (!output.is_open()) {
            std::cerr << "can't open " << args.output << '\n';
            exit(EXIT_FAILURE);
        }
        if (args.command == ENUM_ENCODE) {
            std::cout << "encoding...\n";
            fflush(stdout);
            encode(dnn, &input, &output);
        } else {
            std::cout << "decoding...\n";
            fflush(stdout);
            decode(dnn, &input, &output);
        }
        input.close();
        output.close();
        std::cout << "output written to " << args.output << '\n';
        // free memory
        free_FC_DNN(dnn);
        free(args.input);
        if (args.hidden_layer_size) free(args.hidden_layer_size);
    }
    exit(EXIT_SUCCESS);
}