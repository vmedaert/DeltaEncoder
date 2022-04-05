#ifndef DELTA_ENCODER_H
#define DELTA_ENCODER_H

#define DEBUG_MODE
/*
* disable this to use variable length encoding on the original data (for comparison of compression rate in report)
*/
#define USE_DELTA

#define PARSE_BUFF_SIZE     8

#define ERR_MISSING_FILE    "ERROR: missing %s file\n"
#define ERR_SYNTAX          "ERROR: invalid syntax\n"

#define CMD_TRAIN           "train"
#define CMD_ENCODE          "encode"
#define CMD_DECODE          "decode"

#define CMD_OPT_INPUT_SIZE  "input_layer_size"
#define CMD_OPT_HIDDEN_SIZE "hidden_layer_size"
#define CMD_OPT_ITER_BOUND  "iteration_bound"
#define CMD_OPT_UPDATE_RATE "update_rate"
#define CMD_OPT_ERR_BOUND   "err_bound"
#define CMD_OPT_LEN_BITS    "len_bits"

#define ENUM_TRAIN          0
#define ENUM_ENCODE         1
#define ENUM_DECODE         2

#define ERR_BOUND           1e-9    // stop iterating when the difference in error is less than this
#define ITERATION_BOUND     1e3     // stop iterating after this many iterations
#define UPDATE_RATE         1

#define INPUT_SIZE          32
#define HIDDEN_LAYER_1_SIZE 24
#define HIDDEN_LAYER_2_SIZE 16
#define LEN_BITS            3

typedef struct {
    char* dnn;
    char** input;
    char* output;
    size_t* hidden_layer_size;
    size_t hidden_layer_count;
    size_t input_layer_size;
    size_t input_c;
    size_t iteration_bound;
    size_t update_rate;
    double err_bound;
    uint8_t command;
    uint8_t len_bits;
} DeltaEncoderArgs;

#endif //DELTA_ENCODER_H