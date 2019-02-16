echo "Downloading transistor dataset..."
url=https://stanford.box.com/shared/static/uu0gsf5fn1vidctx8zfpbi4nu9r69hfk.xz
data_tar=transistor_dataset.tar.xz

if type curl &>/dev/null; then
    curl -RL --retry 3 -C - $url -o $data_tar
elif type wget &>/dev/null; then
    wget -N $url -O $data_tar
fi

echo "Unpacking transistor dataset..."
tar vxf $data_tar -C data

echo "Deleting $data_tar..."
rm $data_tar

echo "Done!"
