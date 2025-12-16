#!/bin/bash

size=1048576
rate=32000

if [ $# -ge 1 ]; then
    size=$1 
fi

dir=data
dev=/dev/ttyACM0
baud=2000000
address=0x00410000
timestamp=$(date +%Y%m%d_%H%M%S)
echo "Reading data from flash..."
esptool.py -p $dev -b $baud read_flash $address $size $dir/data_dump_$timestamp.bin
if [ $? -ne 0 ]; then
    echo "Error: Failed to read data from flash"
    exit 1
fi

echo "For wav conversion run:"
echo "python3 bin2wav.py $dir/data_dump_$timestamp.bin $dir/output_$timestamp.wav -r [rate] -C [crc32]"
echo "data_dump file full path: " `pwd`"/$dir/data_dump_$timestamp.bin"
echo "Done"