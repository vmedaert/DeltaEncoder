#ifndef BIT_HANDLER_H
#define BIT_HANDLER_H

#include <stdint.h>
#include <iostream>

//#define DEBUG_BIT_HANDLER

#define MOST_SIG_BIT_64 0x8000000000000000

typedef struct {
    std::ofstream* output;
    uint64_t buff;
    uint8_t free_bits;
    bool big_endian;
} BitWriter;

typedef struct {
    std::ifstream* input;
    uint8_t buff;
    uint8_t buff_bits;
    bool big_endian;
} BitReader;

bool big_endian();

void bitstring(char* str, char* bytes, size_t size, bool big_endian);

void init_bit_writer(BitWriter* writer, std::ofstream* output);

void init_bit_reader(BitReader* reader, std::ifstream* input);

/*
* write bits to output stream
* make sure all bits that should not be written are 0!
*/
void write_bits(BitWriter* writer, uint64_t* bits, uint8_t count);

/*
* write all buffered bits
* the last byte has trailing 1 bits as padding!
*/
void flush_bits(BitWriter* writer);

/*
* read bits from input stream
*/
uint64_t read_bits(BitReader* reader, uint8_t count);

#endif //BIT_HANDLER_H