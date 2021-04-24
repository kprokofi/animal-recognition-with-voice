git clone --depth 1 https://github.com/tensorflow/models
cd $PWD/"models/research/"
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .