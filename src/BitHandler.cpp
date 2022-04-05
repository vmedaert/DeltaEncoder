#include "BitHandler.h"
#include <fstream>
#include <iostream>

#ifdef DEBUG_BIT_HANDLER
static char bitstring_buff[sizeof(uint64_t)*8 + 1];
#endif

bool big_endian() {
    uint64_t x = 1;
    return ((char*) &x)[7] & 0x1;
}

void bitstring(char* str, char* bytes, size_t size, bool big_endian) {
    uint8_t mask;
    if (big_endian) {
        for (size_t i=0; i<size; i++) {
            mask = 0x80;
            for (size_t j=0; j<8; j++) {
                *str = (bytes[i] & mask) ? '1' : '0';
                str++;
                mask >>= 1;
            }
        }
    } else {
        for (int i = size - 1; i>=0; i--) {
            mask = 0x80;
            for (size_t j=0; j<8; j++) {
                *str = (bytes[i] & mask) ? '1' : '0';
                str++;
                mask >>= 1;
            }
        }
    }
    *str = '\0';
}

void init_bit_writer(BitWriter* writer, std::ofstream* output) {
    writer->free_bits = 64;
    writer->buff = 0;
    writer->output = output;
    writer->big_endian = big_endian();
}

void init_bit_reader(BitReader* reader, std::ifstream* input) {
    reader->buff_bits = 0;
    reader->buff = 0;
    reader->input = input;
    reader->big_endian = big_endian();
}

/*
* write bits to output stream
* make sure all bits that should not be written are 0!
*/
void write_bits(BitWriter* writer, uint64_t* bits, uint8_t count) {
    uint8_t byte_count;
    #ifdef DEBUG_BIT_HANDLER
    bitstring(bitstring_buff, (char*) bits, sizeof(uint64_t), writer->big_endian);
    std::cout << "input  = " << bitstring_buff << '\n';
    #endif
    if (count > 64) count = 64;
    if (count > writer->free_bits) {
        byte_count = (64 - writer->free_bits) >> 3;
        // allign bits to be written
        writer->buff <<= writer->free_bits;
        if (writer->big_endian) {
            writer->output->write((char*) &(writer->buff), byte_count);
            #ifdef DEBUG_BIT_HANDLER
            bitstring(bitstring_buff, (char*) &(writer->buff), byte_count, writer->big_endian);
            std::cout << "write    " << bitstring_buff << '\n';
            #endif
        } else {
            for (int i=7; i > 7 - byte_count; i--) {
                writer->output->write(&(((char*) &(writer->buff))[i]), 1);
                #ifdef DEBUG_BIT_HANDLER
                bitstring(bitstring_buff, &(((char*) &(writer->buff))[i]), 1, writer->big_endian);
                std::cout << "write    " << bitstring_buff << '\n';
                #endif
            }
        }
        // allign least significant buffered bit
        writer->buff >>= writer->free_bits;
        writer->free_bits += byte_count * 8;
        if (count > writer->free_bits) {
            // make free bits least significant
            writer->buff <<= writer->free_bits;
            // fill buffer
            writer->buff |= *bits >> (count - writer->free_bits);
            // write entire buffer
            if (writer->big_endian) {
                writer->output->write((char*) &(writer->buff), sizeof(uint64_t));
                #ifdef DEBUG_BIT_HANDLER
                bitstring(bitstring_buff, (char*) &(writer->buff), sizeof(uint64_t), writer->big_endian);
                std::cout << "write    " << bitstring_buff << '\n';
                #endif
            } else {
                for (int i=7; i>=0; i--) {
                    writer->output->write(&(((char*) &(writer->buff))[i]), 1);
                    #ifdef DEBUG_BIT_HANDLER
                    bitstring(bitstring_buff, &(((char*) &(writer->buff))[i]), 1, writer->big_endian);
                    std::cout << "write    " << bitstring_buff << '\n';
                    #endif
                }
            }
            // buffer leftover bits
            writer->buff = *bits;
            writer->free_bits += 64 - count;
        } else {
            // buffer bits
            writer->buff <<= count;
            writer->buff |= *bits;
            writer->free_bits -= count;
        }
    } else {
        // buffer bits
        writer->buff <<= count;
        writer->buff |= *bits;
        writer->free_bits -= count;
    }
    #ifdef DEBUG_BIT_HANDLER
    bitstring(bitstring_buff, (char*) &(writer->buff), sizeof(uint64_t), big_endian());
    std::cout << "buffer = " << bitstring_buff << "\n\n";
    #endif
}

/*
* write all buffered bits
* the last byte has trailing 1 bits as padding!
*/
void flush_bits(BitWriter* writer) {
    uint8_t byte_count = (64 - writer->free_bits) >> 3;
    uint8_t pad_count = 8 - 64 + writer->free_bits + byte_count * 8;
    if (pad_count > 0) {
        byte_count++;
        // add padding bits
        writer->buff <<= pad_count;
        writer->buff |= (UINT64_MAX << pad_count) ^ UINT64_MAX;
        writer->free_bits -= pad_count;
    }
    // allign bits to be written
    writer->buff <<= writer->free_bits;
    if (writer->big_endian) {
        writer->output->write((char*) &(writer->buff), byte_count);
        #ifdef DEBUG_BIT_HANDLER
        bitstring(bitstring_buff, (char*) &(writer->buff), byte_count, writer->big_endian);
        std::cout << "write    " << bitstring_buff << '\n';
        #endif
    } else {
        for (int i=7; i > 7 - byte_count; i--) {
            writer->output->write(&(((char*) &(writer->buff))[i]), 1);
            #ifdef DEBUG_BIT_HANDLER
            bitstring(bitstring_buff, &(((char*) &(writer->buff))[i]), 1, writer->big_endian);
            std::cout << "write    " << bitstring_buff << '\n';
            #endif
        }
    }
    writer->free_bits = 64;
}

/*
* read bits from input stream
*/
uint64_t read_bits(BitReader* reader, uint8_t count) {
    uint64_t bits;
    #ifdef DEBUG_BIT_HANDLER
    printf("count =  %d\n", count);
    bitstring(bitstring_buff, (char*) &(reader->buff), sizeof(uint8_t), reader->big_endian);
    std::cout << "buffer = " << bitstring_buff << '\n';
    #endif
    if (count > 64) count = 64;
    if (reader->buff_bits < count) {
        // read entire buffer
        bits = reader->buff & (UINT64_MAX ^ (UINT64_MAX << reader->buff_bits));
        count -= reader->buff_bits;
        // read bytes from input stream
        while (count > 7) {
            bits <<= 8;
            reader->input->read(&(((char*) &bits)[(reader->big_endian ? 7 : 0)]), 1);
            count -= 8;
        }
        // read one byte from input stream and buffer the unused bits
        if (count > 0) {
            reader->input->read((char*) &(reader->buff), 1);
            bits <<= count;
            bits |= reader->buff >> (8 - count);
            reader->buff_bits = 8 - count;
        } else {
            reader->buff_bits = 0;
        }
    } else {
        // copy buffer and mask out unwanted bits
        bits = (reader->buff >> (reader->buff_bits - count)) & (UINT64_MAX ^ (UINT64_MAX << count));
        reader->buff_bits -= count;
    }
    #ifdef DEBUG_BIT_HANDLER
    bitstring(bitstring_buff, (char*) &bits, sizeof(uint64_t), reader->big_endian);
    std::cout << "output = " << bitstring_buff << "\n\n";
    #endif
    return bits;
}