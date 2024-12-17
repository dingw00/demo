from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import os
import random
import warnings
import PIL.Image as Image

from .transforms import *
from .test_utils import worker_seed_set

DATADIR = "Dataset"

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.root = os.path.dirname(list_path)
        self.class_names = load_classes(os.path.join(self.root, "classes.names"))

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = os.path.join(self.root, self.img_files[index % len(self.img_files)].rstrip())

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = os.path.join(self.root, self.label_files[index % len(self.img_files)].rstrip())

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_dataset(dataset_name, subset="train", img_size=None, batch_size=32, shuffle=None, num_workers=0,
                 multiscale=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale: Scale images to different sizes randomly
    :type multiscale: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    if shuffle is None:
        shuffle = subset == "train"
    
    
    if dataset_name == "coco128":
        classes= 80
        if img_size is None:
            img_size=416
        data_dir = os.path.join(DATADIR, "coco128")
        # Define transformations
        train_tfs = tfs.Compose([AbsoluteLabels(), DefaultAug(), PadSquare(),
                                 RelativeLabels(), ToTensor()])
        val_tfs = tfs.Compose([AbsoluteLabels(), PadSquare(), RelativeLabels(),
                               ToTensor()])
        # Load dataset
        if subset == "train":
            train_cfg_path = os.path.join(data_dir, "train.txt")
            data_set = ListDataset(train_cfg_path,
                                    img_size=img_size,
                                    multiscale=multiscale,
                                    transform=train_tfs)
        
        elif subset == "val" or subset == "test":
            valid_cfg_path = os.path.join(data_dir, "valid.txt")        
            data_set = ListDataset(
                valid_cfg_path,
                img_size=img_size,
                multiscale=multiscale,
                transform=val_tfs)
            
        data_loader = DataLoader(data_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 collate_fn=data_set.collate_fn,
                                 worker_init_fn=worker_seed_set)
    
    else:
        if dataset_name == "railway_track_fault_detection":
            if img_size is None:
                img_size = 448
            data_dir = os.path.join(DATADIR, "railway_track_fault_detection")
            # Define transformations
            train_tfs = test_tfs = val_tfs = tfs.Compose([tfs.Resize((img_size, img_size)), tfs.ToTensor()])
        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented.")
        
        # Load dataset
        if subset == "train":
            data_set = ImageFolder(root=os.path.join(data_dir, "train"), transform=train_tfs)
        elif subset == "val":
            data_set = ImageFolder(root=os.path.join(data_dir, "val"), transform=val_tfs)
        elif subset == "test":
            data_set = ImageFolder(root=os.path.join(data_dir, "test"), transform=test_tfs)

        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_set, data_loader