
Edge-removal Voxel Cloud Connectivity Segmentation
=========================================================

* difference from VCCS
  - voxels at corners or edges are not used for super voxel clustering because their normal is not stable.


PreRequreied
----------------

* Open3D(v0.12)

* nlohmann json
```
wget https://raw.githubusercontent.com/nlohmann/json/v3.9.1/single_include/nlohmann/json.hpp
```


Build
--------

```
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="<Open3D_install_path>/lib/cmake/" ..
make
```


