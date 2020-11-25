ls ### to build ncnn from source
````
sudo apt install git cmake libprotobuf-dev \
                    protobuf-compiler libvulkan-dev  
                    libopencv-dev libpthread-stubs0-dev

sudo apt install libopencv-dev
sudo apt-get install libprotobuf-dev protobuf-compiler
cmake ..
make -j$(nproc)
make install
````


Include dir and lib dir of ncnn is located in "install" dir

Convert from Mxnet to ncnn
```
./mxnet2ncnn mnet.25-symbol.json mnet.25-0000.params face_detection.param face_detection.bin
```