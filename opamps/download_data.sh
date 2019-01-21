echo "Downloading op-amp dataset..."
url=https://stanford.box.com/shared/static/p951ah47t0sm676qn4cwjicmvoyjvxao.xz
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
