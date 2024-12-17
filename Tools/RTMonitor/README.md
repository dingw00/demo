# monitoring-interface

## Requirements

```
pip install -r requirements.txt
```

To install Detectron2, please follow [here](https://detectron2.readthedocs.io/tutorials/install.html).

## Dataset Preparation

We use [Fiftyone](https://docs.voxel51.com) library to load and visualize datasets. 

BDD100k, COCO, KITTI and OpenImage can be loaded directly through [Fiftyone Datasets Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html?highlight=zoo).

For other datasets, such as NuScene can be loaded manually via the following simple pattern:

```python
import fiftyone as fo

# A name for the dataset
name = "my-dataset"

# The directory containing the dataset to import
dataset_dir = "/path/to/dataset"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)
```

The custom dataset folder should have the following structure:

```
 └── /path/to/dataset
     |
     ├── Data
     └── labels.json
```

Notice that the annotation file `labels.json` should be prepared in COCO format.