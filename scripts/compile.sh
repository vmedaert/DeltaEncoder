#!/usr/bin/env sh

DIR=$(echo $0 | sed 's:/[^/]*$::')/..
SRC=${DIR}/src/
DST=${DIR}/bin/

echo "Compiling..."
g++ -O3 -Wall -Werror ${SRC}NeuralNetwork.cpp ${SRC}BitHandler.cpp ${SRC}DeltaEncoder.cpp -o ${DST}DeltaEncoder.exe 
echo "done"
exit 0