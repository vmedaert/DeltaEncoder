#!/usr/bin/env sh

DIR=$(echo $0 | sed 's:/[^/]*$::')/..
TRN=${DIR}/train/
TMP=${DIR}/temp/
BIN=${DIR}/bin/
DNN=${DIR}/neural_networks/

LEN_BITS=4

${BIN}DeltaEncoder.exe --len_bits=${LEN_BITS} encode $1 $2 ${TMP}output.bin > ${TMP}encoder_log.txt
COMPRESSION_RATE=$(python -c "print($(stat -c%s "$2").0/$(stat -c%s "${TMP}output.bin"))")
echo "achieved a compression rate of $COMPRESSION_RATE"
${BIN}DeltaEncoder.exe --len_bits=${LEN_BITS} decode $1 ${TMP}output.bin ${TMP}reconstructed_input.bin > ${TMP}decoder_log.txt
diff -s $2 ${TMP}reconstructed_input.bin
diff -yTt --strip-trailing-cr --width=131 ${TMP}encoder_log.txt ${TMP}decoder_log.txt > ${TMP}diff_log.txt
exit 0