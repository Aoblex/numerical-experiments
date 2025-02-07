import torchvision
import numpy as np
from typing import Callable
from PIL import Image
import os
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from torchvision import transforms
from sklearn.decomposition import PCA

class OT:
    def __init__(self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray):
        """Initialize the OT problem"""
        self.source = source
        self.target = target
        self.cost_matrix = cost_matrix
    
    def __str__(self):
        source_shape = self.source.shape
        target_shape = self.target.shape
        cost_matrix_shape = self.cost_matrix.shape
        return f"Source shape: {source_shape}, Target shape: {target_shape}, Cost matrix shape: {cost_matrix_shape}"

class SinkhornOT(OT):
    def __init__(self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray,
                 reg: float = 0.01):
        """Initialize the Sinkhorn OT problem"""
        super().__init__(source, target, cost_matrix)
        self.reg = reg


class MnistOT:
    n, n_flattend = 28, 784
    def __init__(self, root: str, train: bool = True, download: bool = True) -> None:
        """Initialize the MNIST dataset"""
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=download,
                                                transform=lambda pic: np.float64(pic).flatten())

    def get_distribution(self, idx: int, eps: float = 0.001) -> np.ndarray:
        """Get the smoothed source/target distribution"""
        digit_np = self.mnist[idx][0]
        dist = digit_np / np.sum(digit_np)
        return (1 - eps) * dist + eps / self.n_flattend
         
    def get_cost_matrix(self, distance: str | Callable = 'l2', normailze: bool = True) -> np.ndarray:
        """Get the cost matrix"""
        if distance == 'l1' or 'cityblock':
            distance = lambda x, y: np.linalg.norm(x - y, ord=1)
        elif distance== 'l2' or 'euclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2)
        elif distance == 'l2_squared' or 'sqeuclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2) ** 2
        else:
            pass

        coordinates = np.array([[i, j] for i in range(MnistOT.n) for j in range(MnistOT.n)])
        cost_matrix = np.zeros((MnistOT.n_flattend, MnistOT.n_flattend))

        for i in range(MnistOT.n_flattend):
            for j in range(MnistOT.n_flattend):
                cost_matrix[i, j] = distance(coordinates[i], coordinates[j])
        
        return cost_matrix / np.max(cost_matrix) if normailze else cost_matrix
    
    def get_SinkhornOT(self, source_idx: int, target_idx: int,
                      distance: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_distribution = self.get_distribution(source_idx)
        target_distribution = self.get_distribution(target_idx)
        cost_matrix = self.get_cost_matrix(distance)
        return SinkhornOT(source_distribution, target_distribution, cost_matrix, reg)
                
class FashionMnistOT:
    n, n_flattend = 28, 784
    def __init__(self, root: str, train: bool = True, download: bool = True):
        """Initialize the Fashion MNIST dataset"""
        self.fashion_mnist = torchvision.datasets.FashionMNIST(root=root, train=train, download=download,
                                                transform=lambda pic: np.float64(pic).flatten())

    def get_distribution(self, idx: int, eps: float = 0.001):
        """Get the smoothed source/target distribution"""
        digit_np = self.fashion_mnist[idx][0]
        dist = digit_np / np.sum(digit_np)
        return (1 - eps) * dist + eps / self.n_flattend
    
    def get_cost_matrix(self, distance: str | Callable = 'l2', normailze: bool = True):
        """Get the cost matrix"""
        if distance == 'l1' or 'cityblock':
            distance = lambda x, y: np.linalg.norm(x - y, ord=1)
        elif distance== 'l2' or 'euclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2)
        elif distance == 'l2_squared' or 'sqeuclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2) ** 2
        else:
            pass

        coordinates = np.array([[i, j] for i in range(FashionMnistOT.n) for j in range(FashionMnistOT.n)])
        cost_matrix = np.zeros((FashionMnistOT.n_flattend, FashionMnistOT.n_flattend))

        for i in range(FashionMnistOT.n_flattend):
            for j in range(FashionMnistOT.n_flattend):
                cost_matrix[i, j] = distance(coordinates[i], coordinates[j])
        
        return cost_matrix / np.max(cost_matrix) if normailze else cost_matrix
    
    def get_SinkhornOT(self, source_idx: int, target_idx: int,
                        distance: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_distribution = self.get_distribution(source_idx)
        target_distribution = self.get_distribution(target_idx)
        cost_matrix = self.get_cost_matrix(distance)
        return SinkhornOT(source_distribution, target_distribution, cost_matrix, reg)

class ImagenetteOT:

    def __init__(self, root: str, split: str = 'train', size: str = 'full', download: bool = True, dim: int = 30):

        """Initialize the Resnet18 model"""
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.eval()
        # The transforms for the Resnet18 model is already defined in
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
        self.resnet18_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

        """Initialize the Imagenette dataset"""
        self.root = root
        self.dim = dim
        # transform: prepare the images for the Resnet18 model
        self.Imagenette = torchvision.datasets.Imagenette(root=root, split=split, size=size, download=download,
                                                          transform=self.resnet18_transforms)
        self._process_Imagenette()
    
    @property
    def class2idx(self):
        return self.Imagenette.class_to_idx

    def _get_layer_output(self, input_tensor: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Given a single picture, get the output of the layer according to the layer name"""
        # if not batched, add a batch dimension
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            for name, module in self.resnet18.named_children():
                input_tensor = module(input_tensor)
                if name == layer_name:
                    return input_tensor.squeeze().numpy()
        
        # if the layer name is not found, return the last layer
        return input_tensor.squeeze().numpy()
    
    def _get_classname(self, idx: int):
        """Get the classname: select the first name"""
        return self.Imagenette.classes[idx][0]
    
    def read_all_class(self):
        """Read the processed Imagenette dataset"""
        save_path = os.path.join(self.root, f'imagenette2-{self.dim}.pkl')
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    def read_processed_class(self, classname: str):
        """Read the processed Imagenette dataset"""
        save_path = os.path.join(self.root, f'imagenette2-{self.dim}.pkl')
        with open(save_path, 'rb') as f:
            return pickle.load(f)[classname]

    def _process_Imagenette(self):
        """Convert the images in each class to a {dim} x 1 vector
        using Resnet18 with IMAGENET1K_V1 weights, then save the vectors"""
        # save the vectors in a dictionary
        save_path = os.path.join(self.root, f'imagenette2-{self.dim}.pkl')
        if os.path.exists(save_path):
            print(f"Processed Imagenette dataset found at {save_path}")
            return

        input_dict = {}
        layer_name = 'avgpool'
        for img, idx in tqdm(self.Imagenette, desc='Processing Imagenette'):
            # img is a 3 x 224 x 224 tensor
            classname = self._get_classname(idx)
            if classname not in input_dict:
                input_dict[classname] = []
            
            converted_img = self._get_layer_output(img, layer_name)
            input_dict[classname].append(converted_img)
        
        # do PCA on the vectors
        pca = PCA(n_components=self.dim)
        for classname, vectors in input_dict.items():
            np_vectors = np.array(vectors)
            input_dict[classname] = pca.fit_transform(np_vectors)
        
        # save the dictionary
        with open(save_path, 'wb') as f:
            pickle.dump(input_dict, f)

    def get_distribution(self, classname: str) -> np.ndarray:
        # vectors is an ndarray of shape (n, dim), where n is the number of images in the class
        vectors = self.read_processed_class(classname)
        n = vectors.shape[0]
        return 1.0 / n * np.ones(n)
    
    def get_cost_matrix(self, source_classname: str, target_classname: str,
                        distance: str | Callable = 'l2', normalize: bool = True) -> np.ndarray:
        """Get the distance of the cost matrix"""
        if distance == 'l1' or 'cityblock':
            distance = lambda x, y: np.linalg.norm(x - y, ord=1)
        elif distance== 'l2' or 'euclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2)
        elif distance == 'l2_squared' or 'sqeuclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2) ** 2
        """Get the source and target vectors"""
        source_vector = self.read_processed_class(source_classname)
        target_vector = self.read_processed_class(target_classname)
        n, m = source_vector.shape[0], target_vector.shape[0]
        """Calculate the cost matrix"""
        cost_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = distance(source_vector[i], target_vector[j])

        return cost_matrix / np.max(cost_matrix) if normalize else cost_matrix
    
    def get_SinkhornOT(self, source_classname: str, target_classname: str,
                       distance: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_distribution = self.get_distribution(source_classname)
        target_distribution = self.get_distribution(target_classname)
        cost_matrix = self.get_cost_matrix(source_classname, target_classname, distance)
        return SinkhornOT(source_distribution, target_distribution, cost_matrix, reg)

class SyntheticOT:

    def __init__(self, n: int = 100, m: int = 100, seed: int = 42):
        """Initialize the synthetic dataset"""
        self.n = n
        self.m = m
        self.rng = np.random.default_rng(seed)
    
    def get_points(self, size: int, spacing: str = 'uniform', low: float = 0.0, high: float = 5.0):
        """Get the points"""
        if spacing == 'uniform':
            return np.linspace(low, high, size)
        elif spacing == 'random':
            return self.rng.uniform(low, high, size)

        return np.linspace(low, high, size)
    
    def get_distribution(self, density_fn: Callable, points: np.ndarray = None):
        """Get the source/target distribution"""
        densities = np.array([density_fn(point) for point in points])
        return densities / np.sum(densities)
    
    def get_cost_matrix(self, distance: str | Callable = 'l2',
                        source_points: np.ndarray = None,
                        target_points: np.ndarray = None,
                        normalize: bool = True):
        """Get the cost matrix"""
        if distance == 'l1' or 'cityblock':
            distance = lambda x, y: np.linalg.norm(x - y, ord=1)
        elif distance== 'l2' or 'euclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2)
        elif distance == 'l2_squared' or 'sqeuclidean':
            distance = lambda x, y: np.linalg.norm(x - y, ord=2) ** 2
        elif distance == 'random':
            cost_matrix = self.rng.uniform(0, 1, (len(source_points), len(target_points)))
            return cost_matrix / np.max(cost_matrix) if normalize else cost_matrix

        cost_matrix = np.zeros((len(source_points), len(target_points)))
        for i in range(len(source_points)):
            for j in range(len(target_points)):
                cost_matrix[i, j] = distance(source_points[i], target_points[j])
        
        return cost_matrix / np.max(cost_matrix) if normalize else cost_matrix
    
    def get_SinkhornOT(self, source_density_fn: Callable, target_density_fn: Callable,
                       source_spacing: str = 'uniform', target_spacing: str = 'uniform',
                       source_low: float = 0.0, source_high: float = 5.0,
                       target_low: float = 0.0, target_high: float = 5.0,
                       distance: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_points = self.get_points(self.n, source_spacing, source_low, source_high)
        target_points = self.get_points(self.m, target_spacing, target_low, target_high)
        source_distribution = self.get_distribution(source_density_fn, source_points)
        target_distribution = self.get_distribution(target_density_fn, target_points)
        cost_matrix = self.get_cost_matrix(distance, source_points, target_points)
        return SinkhornOT(source_distribution, target_distribution, cost_matrix, reg)
