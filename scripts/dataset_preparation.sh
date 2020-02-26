public_dataset_path="/path/to/kitti/official/dataset" # e.g. "/data/public_dataset/kitti/object"

cd ../data/KITTI/object/||exit
rm -rf training testing
mkdir training
mkdir testing

cd training||exit
ln -s $public_dataset_path/data_object_calib/training/calib/ .
ln -s $public_dataset_path/data_object_image_2/training/image_2/ .
ln -s $public_dataset_path/data_object_image_3/training/image_3/ .
ln -s $public_dataset_path/data_object_label_2/training/label_2/ .
ln -s $public_dataset_path/data_object_velodyne/training/velodyne/ .

cd ../testing||exit
ln -s $public_dataset_path/data_object_calib/testing/calib/ .
ln -s $public_dataset_path/data_object_image_2/testing/image_2/ .
ln -s $public_dataset_path/data_object_image_3/testing/image_3/ .
ln -s $public_dataset_path/data_object_velodyne/testing/velodyne/ .
