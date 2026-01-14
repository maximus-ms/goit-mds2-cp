#!/bin/bash

crc32=0
size=$(( 1024*256 ))
rate=16000
width=16

if [ $# -ge 1 ]; then
    crc32=$1
fi
if [ $# -ge 2 ]; then
    size=$2
fi
if [ $# -ge 3 ]; then
    rate=$3
fi
if [ $# -ge 4 ]; then
    width=$4
fi

dir=data
dev=/dev/ttyACM0
baud=2000000
address=0x00410000
timestamp="_$(date +%Y%m%d_%H%M%S)"
raw_data_file=$dir/data_dump$timestamp.bin
wav_file=$dir/output$timestamp.wav
echo "Reading data from flash..."
esptool -p $dev -b $baud read-flash $address $size $raw_data_file
if [ $? -ne 0 ]; then
    echo "Error: Failed to read data from flash"
    exit 1
fi
echo "Converting to WAV..."
python3 bin2wav.py $raw_data_file $wav_file -c 2 -w $width -r $rate -C $crc32 
if [ $? -ne 0 ]; then
    echo "Error: Failed to convert to WAV"
    exit 1
fi
echo "Done"
