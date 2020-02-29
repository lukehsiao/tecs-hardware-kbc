echo "Downloading op-amp dataset..."
url=https://stanford.box.com/shared/static/xszeoh0o36j2d7jn05mbojhnyg61nfr3.xz
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
url=https://stanford.box.com/shared/static/568p14l4zxnfbgtrtzpm8wr2ynp2328l.xz
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
