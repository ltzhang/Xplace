#!/bin/bash

echo "============================================================"
echo " If wget fails or downloads are too slow, you can manually"
echo " download the .tar.gz files from OneDrive links in README.md"
echo " and place them in this directory before re-running this script."
echo "============================================================"

mkdir -p raw

echo "=== Getting ispd2005 ==="
if [ -d "raw/ispd2005" ]; then
    echo "raw/ispd2005 already exists, skipping."
elif [ ! -f "ispd2005.tar.gz" ]; then
    echo "Error: ispd2005.tar.gz not found. Please download it manually (see README.md)." >&2
    exit 1
else
    tar xvzf ispd2005.tar.gz
    mv ispd2005/ raw/
fi

echo "=== Getting ispd2006 ==="
if [ -d "raw/ispd2006" ]; then
    echo "raw/ispd2006 already exists, skipping."
elif [ ! -f "ispd2006.tar.gz" ]; then
    echo "Error: ispd2006.tar.gz not found. Please download it manually (see README.md)." >&2
    exit 1
else
    tar xvzf ispd2006.tar.gz
    mv ispd2006/ raw/
fi

echo "=== Getting mms ==="
if [ -d "raw/mms" ]; then
    echo "raw/mms already exists, skipping."
elif [ ! -f "mms.tar.gz" ]; then
    echo "Error: mms.tar.gz not found. Please download it manually (see README.md)." >&2
    exit 1
else
    tar xvzf mms.tar.gz
    mv mms/ raw/
fi

echo "=== Getting ispd2015 ==="
if [ -d "raw/ispd2015" ]; then
    echo "raw/ispd2015 already exists, skipping."
elif [ ! -f "ispd2015.tar.gz" ]; then
    echo "Error: ispd2015.tar.gz not found. Please download it manually (see README.md)." >&2
    exit 1
else
    tar xvzf ispd2015.tar.gz
    mv ispd2015/ raw/
fi
echo "=== Preprocessing ispd2015 to generate ispd2015_fix ==="
python fix_ispd2015_route.py

echo "=== Getting iccad2015 ==="
if [ -d "raw/iccad2015" ]; then
    echo "raw/iccad2015 already exists, skipping."
elif [ ! -f "iccad2015.tar.gz" ]; then
    echo "Error: iccad2015.tar.gz not found. Please download it manually (see README.md)." >&2
    exit 1
else
    tar xvzf iccad2015.tar.gz
    mv iccad2015/ raw/
fi

echo "=== Getting iccad2019 ==="
if [ -d "raw/iccad2019" ]; then
    echo "raw/iccad2019 already exists, skipping."
elif [ ! -f "iccad2019.tar.gz" ]; then
    echo "Error: iccad2019.tar.gz not found. Please download it manually (see README.md)." >&2
    exit 1
else
    tar xvzf iccad2019.tar.gz
    mv iccad2019/ raw/
fi

echo "=== Getting ispd2018 ==="
mkdir -p raw/ispd2018
for i in {1..10}
do
    wget --no-check-certificate https://www.ispd.cc/contests/18/ispd18_test$i.tgz
    tar xvzf ispd18_test$i.tgz
    rm -rf ispd18_test$i.tgz
    mv ispd18_test$i/ raw/ispd2018/
done

echo "=== Getting ispd2019 ==="
mkdir -p raw/ispd2019
for i in {1..10}
do
    wget --no-check-certificate https://www.ispd.cc/contests/19/benchmarks/ispd19_test$i.tgz
    tar xvzf ispd19_test$i.tgz
    rm -rf ispd19_test$i.tgz
    mv ispd19_test$i/ raw/ispd2019/
done
python remove_fence_in_ispd19_test5.py

# echo "=== (Optional) Converting raw design to torch data ==="
# python convert_design_to_torch_data.py --dataset ispd2005
# python convert_design_to_torch_data.py --dataset ispd2015_fix
# python convert_design_to_torch_data.py --dataset iccad2019
# python convert_design_to_torch_data.py --dataset ispd2019
