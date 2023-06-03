#!/bin/bash
sed -i '2s/2/4/g' def.h
make
echo "----------------vector width 4"
./myexp -s 10000

sed -i '2s/4/8/g' def.h
make
echo "----------------vector width 8"
./myexp -s 10000


sed -i '2s/8/16/g' def.h
make
echo "----------------vector width 16"
./myexp -s 10000