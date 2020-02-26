cd ..
cd libs/pointnet2_lib/pointnet2/||exit
rm -rf build/ dist/ pointnet2.egg-info/
python setup.py install

cd ../../pointrcnn/iou3d/||exit
rm -rf build/ dist/ iou3d.egg-info/
python setup.py install

cd ../roipool3d/||exit
rm -rf build/ dist/ roipool3d.egg-info/
python setup.py install

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
cd ../../GANet||exit
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib

cd ../sync_bn||exit
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib
