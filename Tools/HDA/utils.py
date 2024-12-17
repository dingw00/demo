import os
import os.path
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import numpy as np
import time
from Tools.HDA.multi_level import multilevel_uniform, greyscale_multilevel_uniform

def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle = True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def save_checkpoint_adv(model,mode,model_dir, epoch):
    path = os.path.join(model_dir, model.name+'_'+mode)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint_adv(model, model_dir,mode):
    path = os.path.join(model_dir, model.name+'_'+mode)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']


def load_checkpoint(model, model_dir,cuda):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch

def get_nearest_oppo_dist(X, y, norm, n_jobs=10):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    p = norm

    def helper(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=p, n_jobs=12).fit(X[y != yi])

    nns = Parallel(n_jobs=n_jobs)(delayed(helper)(yi) for yi in np.unique(y))
    ret = np.zeros(len(X))
    for yi in np.unique(y):
        dist, _ = nns[yi].kneighbors(X[y == yi], n_neighbors=1)
        ret[np.where(y == yi)[0]] = dist[:, 0]

    return nns, ret

def filter_celeba(dataset):
    # drop unrelated attr
    attr = dataset.attr
    attr_names = dataset.attr_names[:40]
    new_attr_names = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    mask_attr = torch.tensor([True if x in new_attr_names else False for x in attr_names]) 
    dataset.attr = attr[:,mask_attr]
    dataset.attr_names = new_attr_names
    # keep only 1 attr instance and drop others
    mask_id = torch.sum(dataset.attr, dim = 1) == 1
    dataset = torch.utils.data.Subset(dataset, torch.where(mask_id)[0])
    return dataset

def cal_robust(x_sample, x_class, model, CUDA, grey_scale,sigma):

    if grey_scale:
        robustness_stat = greyscale_multilevel_uniform
    else:
        robustness_stat = multilevel_uniform

    # sigma = 0.1
    rho = 0.1
    debug= True
    stats=False
    count_particles = 1000
    count_mh_steps = 200

    print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)

    def prop(x):
      y = model(x)
      y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff #.max(dim=1)

    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = robustness_stat(prop, x_sample, sigma, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                              count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end - start) / 60} minutes...')

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)
      print('---------------------------------')

    return lg_p


def mutation(x_seed, adv_images, eps, p):
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.003)
    mask = torch.empty_like(adv_images).uniform_() > p 
    delta[mask] = 0.0
    delta = adv_images + delta - x_seed
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0, max=1).detach()
    return adv_images


def cal_dist(x, x_a, model):
    model.eval()
    act_a = model(x_a)[0]
    act = model(x)[0]
    act_a = torch.flatten(act_a, start_dim = 1)
    act = torch.flatten(act, start_dim = 1)
    mse = calculate_fid(act, act_a)
    return mse

# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*torch.log2(q[i]) for i in range(len(p))])

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    images = []
    labels = []
    idx = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            
            # Extracting id (a) and true label (b) from the filename
            file_parts = os.path.splitext(filename)[0].split('_')
            if (len(file_parts) >= 2) and (file_parts[-1] != "failed"):
                img_id, true_label = map(int, file_parts[:2])
                
                # Load image and append to the list
                img = Image.open(img_path).convert('RGB')
                transform = transforms.ToTensor()
                img = transform(img)
            
                images.append(img)
                labels.append(torch.tensor(true_label))
                idx.append(img_id)

    return idx, images, labels

def calculate_and_display_image_norms(path_A, path_B, norm_type):
    images_A = sorted([os.path.join(path_A, file) for file in os.listdir(path_A) if file.endswith(('png', 'jpg', 'jpeg'))])
    images_B = sorted([os.path.join(path_B, file) for file in os.listdir(path_B) if file.endswith(('png', 'jpg', 'jpeg'))])

    transform = transforms.ToTensor()

    def image_to_tensor(image_path):
        image = Image.open(image_path).convert('RGB')
        return transform(image)

    def display_images(images, title):
        rows = 2
        cols = len(images) // rows + (len(images) % rows > 0)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5))
        fig.suptitle(title, fontsize=20)
        axs = axs.flatten()
        for img, ax in zip(images, axs):
            ax.imshow(Image.open(img))
            ax.axis('off')
        for ax in axs[len(images):]:
            ax.axis('off')
        plt.show()

    differences = []
    for img_A_path, img_B_path in zip(images_A, images_B):
        tensor_A = image_to_tensor(img_A_path)
        tensor_B = image_to_tensor(img_B_path)
        
        l_norm = torch.norm(tensor_A - tensor_B, p=norm_type).item()
        differences.append(l_norm)

    average_difference = np.mean(differences)
    print(f"Average L{norm_type} norm: {average_difference}")

    display_images(images_A, 'Origin Images')
    display_images(images_B, 'AE Images')


    




