echo "Downloading op-amp dataset..."
url=https://zenodo.org/record/4558344/files/opamp_dataset.tar.xz?download=1
data_tar=opamp_dataset.tar.xz

if type curl &>/dev/null; then
    curl -RL --retry 3 -C - $url -o $data_tar
elif type wget &>/dev/null; then
    wget -N $url -O $data_tar
fi

echo "Unpacking op-amp dataset..."
tar vxf $data_tar -C data

echo "Deleting $data_tar..."
rm $data_tar

echo "Done!"

echo "Downloading mouser op-amp dataset..."
url=https://zenodo.org/record/4558344/files/mouser_opamps.tar.xz?download=1
data_tar=mouser_opamps.tar.xz

if type curl &>/dev/null; then
    curl -RL --retry 3 -C - $url -o $data_tar
elif type wget &>/dev/null; then
    wget -N $url -O $data_tar
fi

echo "Unpacking mouser op-amp dataset..."
tar vxf $data_tar -C data

echo "Deleting $data_tar..."
rm $data_tar

echo "Done!"
