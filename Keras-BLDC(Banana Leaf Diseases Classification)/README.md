# Keras-BLDC(Banana Leaf Diseases Classification)
re-implementation of 'A Deep Learning-based Approach for Banana Leaf Diseases Classification' using Keras
 
## usage
1. download PlantVillage Dataset
    Unfortunately, PlantVillage Dataset is not available now. <br />
    As an alternative, download the different dataset from [[PlantVillage-Dataset]](https://github.com/spMohanty/PlantVillage-Dataset) <br />
    However, PlantVillage Dataset in above repo does not have a data related Banana Leaf <br />
    So, I alternatively use Apple Leaf(containing in above repo) instead of Banana Leaf <br />

2. Configure folder structure
    Copy Apple__XX folder from PlantVillage-Dataset/raw/color to path/to/dataset/BLDC_color/raw <br />
    similarly copy Apple__XX folder from PlantVillage-Dataset/raw/garyscale to path/to/dataset/BLDC_grayscale/raw <br />
     
    ```
    path/to/dataset
    ├── BLDC_color
    │   └── raw
    │       └── Apple___Apple_scab
    │           └── 0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG
    │           └── ...
    │       └── Apple___Black_rot
    │           └── ...
    │       └── Apple___Cedar_apple_rust
    │           └── ...
    │       └── Apple___healthy
    │           └── ...
    ├── BLDC_grayscale
    │   └── raw
    │       └── Apple___Apple_scab
    │           └── 0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG
    │           └── ...
    │       └── Apple___Black_rot
    │           └── ...
    │       └── Apple___Cedar_apple_rust
    │           └── ...
    │       └── Apple___healthy
    │           └── ...
    ```

3. Split dataset(train/test)
Modify path of src_dir and dst_dir in split_dataset.py <br />
and, run split_dataset.py python script

```commandline
    python split_dataset.py
```

this script will separate 'path/to/dataset/BLDC_XXX/raw' into train and test set

4. Training Model
Modify path of data_dir in train.py <br />
and, run train.py python script

```commandline
    python train.py
```

## Note
in this repo, I re-implement 'A Deep Learning-based Approach for Banana Leaf Diseases Classification' paper using Keras. <br />
I alternatively using Challenge Dataset from [[PlantVillage-Dataset]](https://github.com/spMohanty/PlantVillage-Dataset) and using Apply Leaf Data because PlantVillage Dataset is not available maybe after 2018. <br />
So, behavior of this implementation is not same with paper...


## Reference
[1] [A Deep Learning-based Approach for Banana Leaf Diseases Classification, 2017](http://btw2017.informatik.uni-stuttgart.de/slidesandpapers/E1-10/paper_web.pdf) <br/>
<!-- [5] []() <br/> -->