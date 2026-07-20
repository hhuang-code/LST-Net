# compile pointops
cd lib/pointops
python setup.py install --home="."

# compile the sub-sampling and knn op
cd lib/nearest_neighbors
python setup.py install --home="."

cd lib/cpp_wrappers
bash compile_wrappers.sh

