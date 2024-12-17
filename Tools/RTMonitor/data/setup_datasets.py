import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import Tools.RTMonitor.data.metadata as metadata
import gdown
import tarfile
import zipfile

def _download(url, save_dir):
    tmp_file_path = os.path.dirname(save_dir)+"/tmp"
    gdown.download(url, tmp_file_path)
    try:
        with tarfile.open(tmp_file_path, "r") as tar:
            tar.extractall()
    except tarfile.ReadError as e:
        print(e)
        with zipfile.ZipFile(tmp_file_path, "r") as zip:
            zip.extractall()

def setup_all_datasets(dataset_dir, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_voc_ood_dataset(dataset_dir)
    setup_kitti_dataset(dataset_dir)
    setup_voc_dataset(dataset_dir)
    setup_coco_dataset(
        dataset_dir,
        image_root_corruption_prefix=image_root_corruption_prefix)
    setup_coco_ood_dataset(dataset_dir)
    setup_openim_odd_dataset(dataset_dir)
    setup_bdd_dataset(dataset_dir)
    setup_coco_ood_bdd_dataset(dataset_dir)
    setup_nu_dataset(dataset_dir)
def setup_nu_dataset(dataset_dir):

    train_image_dir = dataset_dir
    train_json_annotations = os.path.join(
        dataset_dir, 'nuimages_v1.0-train.json')
    register_coco_instances(
        "nu_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "nu_custom_train").thing_classes = metadata.NU_THING_CLASSES
    MetadataCatalog.get(
        "nu_custom_train").thing_dataset_id_to_contiguous_id = metadata.NU_THING_DATASET_ID_TO_CONTIGUOUS_ID
    
    test_image_dir = dataset_dir
    test_json_annotations = os.path.join(
        dataset_dir, 'nuimages_v1.0-val.json')

    register_coco_instances(
        "nu_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "nu_custom_val").thing_classes = metadata.NU_THING_CLASSES
    MetadataCatalog.get(
        "nu_custom_val").thing_dataset_id_to_contiguous_id = metadata.NU_THING_DATASET_ID_TO_CONTIGUOUS_ID
    
def setup_voc_ood_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=1WWikOdHu5CMsagds3ivE5cHJEvUnvHjB&export=download&authuser=0&confirm=t&uuid=79a43a01-71b0-4fa5-8b2e-da09328d5059&at=APZUnTU44Ta7nsuP97Tgisnq104n%3A1710011734739"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    test_image_dir = os.path.join(dataset_dir, 'data')

    test_json_annotations = os.path.join(
        dataset_dir, 'labels.json')

    register_coco_instances(
        "voc_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_ood_val").thing_classes = metadata.VOC_OOD_THING_CLASSES
    MetadataCatalog.get(
        "voc_ood_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain
    
def setup_kitti_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=1e9FmPUPzRBHN0OTQndqARLQJ_F1QRHu0&export=download&authuser=0&confirm=t&uuid=8168f2ca-cc02-4d7e-8e5c-55da3417debb&at=APZUnTXR__U8x9yzlRa8h3AhKCb3%3A1710013134243"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    train_image_dir = os.path.join(dataset_dir, 'train_split/data')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'val_split/data')

    train_json_annotations = os.path.join(
        dataset_dir, 'train_split/labels.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_split/labels.json')

    register_coco_instances(
        "kitti_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "kitti_custom_train").thing_classes = metadata.KITTI_THING_CLASSES
    MetadataCatalog.get(
        "kitti_custom_train").thing_dataset_id_to_contiguous_id = metadata.KITTI_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "kitti_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "kitti_custom_val").thing_classes = metadata.KITTI_THING_CLASSES
    MetadataCatalog.get(
        "kitti_custom_val").thing_dataset_id_to_contiguous_id = metadata.KITTI_THING_DATASET_ID_TO_CONTIGUOUS_ID
def setup_coco_dataset(dataset_dir, image_root_corruption_prefix=None):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    url = "https://drive.usercontent.google.com/download?id=16dBADw9xXQz4AGTrScQCvkcilhYI7nMo&export=download&authuser=0&confirm=t&uuid=ff68c37e-d64f-4baa-9b5e-230965871564&at=APZUnTW5qlEZqkay0ZgHaQPJNr8H%3A1710013172964"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    train_image_dir = os.path.join(dataset_dir, 'train2017')

    if image_root_corruption_prefix is not None:
        test_image_dir = os.path.join(
            dataset_dir, 'val2017' + image_root_corruption_prefix)
    else:
        test_image_dir = os.path.join(dataset_dir, 'val2017')

    train_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openim_dataset(dataset_dir):
    """
    sets up openimages dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    url = "https://drive.usercontent.google.com/download?id=1sQ6ignp0N-g4DerzZWcbiB0kBtK97a9d&export=download&authuser=0&confirm=t&uuid=06a7196c-91bb-4510-9d55-4fdb2b676e36&at=APZUnTVLCwJQ-m0eMg8ldM0mVyWc%3A1710013233752"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)
    # import ipdb; ipdb.set_trace()
    test_image_dir = os.path.join(dataset_dir, 'images')

    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openim_odd_dataset(dataset_dir):
    """
    sets up openimages out-of-distribution dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    url = "https://drive.usercontent.google.com/download?id=1sQ6ignp0N-g4DerzZWcbiB0kBtK97a9d&export=download&authuser=0&confirm=t&uuid=06a7196c-91bb-4510-9d55-4fdb2b676e36&at=APZUnTVLCwJQ-m0eMg8ldM0mVyWc%3A1710013233752"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    test_image_dir = os.path.join(dataset_dir, 'ood_classes_rm_overlap', 'images')

    test_json_annotations = os.path.join(
        dataset_dir, 'ood_classes_rm_overlap', 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_voc_id_dataset(dataset_dir):
    url = "https://drive.usercontent.google.com/download?id=1WWikOdHu5CMsagds3ivE5cHJEvUnvHjB&export=download&authuser=0&confirm=t&uuid=79a43a01-71b0-4fa5-8b2e-da09328d5059&at=APZUnTU44Ta7nsuP97Tgisnq104n%3A1710011734739"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)
    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train_id",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train_id").thing_classes = metadata.VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train_id").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain

    register_coco_instances(
        "voc_custom_val_id",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val_id").thing_classes = metadata.VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val_id").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain
def setup_bdd_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'images/100k/train')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'images/100k/val')

    train_json_annotations = os.path.join(
        dataset_dir, 'train_bdd_converted.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_bdd_converted.json')
    
    register_coco_instances(
        "bdd_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "bdd_custom_train").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_custom_train").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "bdd_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "bdd_custom_val").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_custom_val").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_voc_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=1WWikOdHu5CMsagds3ivE5cHJEvUnvHjB&export=download&authuser=0&confirm=t&uuid=79a43a01-71b0-4fa5-8b2e-da09328d5059&at=APZUnTU44Ta7nsuP97Tgisnq104n%3A1710011734739"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "voc_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_ood_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=16dBADw9xXQz4AGTrScQCvkcilhYI7nMo&export=download&authuser=0&confirm=t&uuid=ff68c37e-d64f-4baa-9b5e-230965871564&at=APZUnTW5qlEZqkay0ZgHaQPJNr8H%3A1710013172964"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'instances_val2017_ood_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_ood_bdd_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=16dBADw9xXQz4AGTrScQCvkcilhYI7nMo&export=download&authuser=0&confirm=t&uuid=ff68c37e-d64f-4baa-9b5e-230965871564&at=APZUnTW5qlEZqkay0ZgHaQPJNr8H%3A1710013172964"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_wrt_bdd_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val_bdd",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val_bdd").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val_bdd").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_ood_train_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=16dBADw9xXQz4AGTrScQCvkcilhYI7nMo&export=download&authuser=0&confirm=t&uuid=ff68c37e-d64f-4baa-9b5e-230965871564&at=APZUnTW5qlEZqkay0ZgHaQPJNr8H%3A1710013172964"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)

    test_image_dir = os.path.join(dataset_dir, 'train2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017_ood.json')

    register_coco_instances(
        "coco_ood_train",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openimages_ood_oe_dataset(dataset_dir):

    url = "https://drive.usercontent.google.com/download?id=1sQ6ignp0N-g4DerzZWcbiB0kBtK97a9d&export=download&authuser=0&confirm=t&uuid=06a7196c-91bb-4510-9d55-4fdb2b676e36&at=APZUnTVLCwJQ-m0eMg8ldM0mVyWc%3A1710013233752"
    if not os.path.exists(dataset_dir):
        _download(url, dataset_dir)
        
    test_image_dir = os.path.join(dataset_dir, 'images')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_oe",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "openimages_ood_oe").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_oe").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID