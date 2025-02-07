import torchvision
import numpy as np
from typing import Callable
import os
from tqdm import tqdm
import pickle
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from sklearn.decomposition import PCA
from .objectives import SinkhornOT
from scipy.stats import expon, norm
from abc import abstractmethod

ROOT = './data'

class BaseOT:

    def __init__(
        self,
    ) -> None:
        self.source_distribution = None
        self.target_distribution = None
        self.cost_matrix = None
        self.description = None

    @abstractmethod
    def _get_distribution(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _get_cost_matrix(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def a(self) -> np.ndarray:
        return self.source_distribution
    
    @property
    def b(self) -> np.ndarray:
        return self.target_distribution
    
    @property
    def M(self) -> np.ndarray:
        return self.cost_matrix


class MnistOT(BaseOT):
    n, n_flattend = 28, 784
    def __init__(
        self,
        source_idx: int = 2,
        target_idx: int = 54698,
        root: str = ROOT,
        reg: float = 0.01,
        distance: str | Callable = 'l2',
    ) -> None:
        """Initialize the MNIST dataset"""
        self.mnist = torchvision.datasets.MNIST(root=root, train=True, download=True,
                                                transform=lambda pic: np.float64(pic).flatten())
        self.distance = distance
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.reg = reg

        self.source_distribution = self._get_distribution(source_idx)
        self.target_distribution = self._get_distribution(target_idx)
        self.cost_matrix = self._get_cost_matrix(distance)
        self.description = f"MNIST.{source_idx}.{target_idx}.norm={distance}.reg={reg}"
    
    def _get_distribution(self, idx: int, eps: float = 0.001) -> np.ndarray:
        """Get the smoothed source/target distribution"""
        digit_np = self.mnist[idx][0]
        dist = digit_np / np.sum(digit_np)
        return (1 - eps) * dist + eps / self.n_flattend
         
    def _get_cost_matrix(self, distance: str | Callable = 'l2') -> None:
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
        
        return cost_matrix / np.max(cost_matrix)
    

class FashionMnistOT(BaseOT):
    n, n_flattend = 28, 784
    def __init__(
        self,
        source_idx: int = 2,
        target_idx: int = 54698,
        root: str = ROOT,
        reg: float = 0.01,
        distance: str | Callable = 'l2',
    ) -> None:
        """Initialize the FashionMNIST dataset"""
        self.fashion_mnist = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,
                                                               transform=lambda pic: np.float64(pic).flatten())
        self.distance = distance
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.reg = reg

        self.source_distribution = self._get_distribution(source_idx)
        self.target_distribution = self._get_distribution(target_idx)
        self.cost_matrix = self._get_cost_matrix(distance)
        self.description = f"FashionMNIST.{source_idx}.{target_idx}.norm={distance}.reg={reg}"
    
    def _get_distribution(self, idx: int, eps: float = 0.001) -> np.ndarray:
        """Get the smoothed source/target distribution"""
        digit_np = self.fashion_mnist[idx][0]
        dist = digit_np / np.sum(digit_np)
        return (1 - eps) * dist + eps / self.n_flattend
    
    def _get_cost_matrix(self, distance: str | Callable = 'l2') -> None:
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
        
        return cost_matrix / np.max(cost_matrix)
    

class ImagenetteOT(BaseOT):
    """
    The Imagenette dataset is a subset of 10 classes from the Imagenet dataset.
    [
        ('tench', 'Tinca tinca'),
        ('English springer', 'English springer spaniel'),
        ('cassette player',),
        ('chain saw', 'chainsaw'),
        ('church', 'church building'),
        ('French horn', 'horn'),
        ('garbage truck', 'dustcart'),
        ('gas pump', 'gasoline pump', 'petrol pump', 'island dispenser'),
        ('golf ball',),
        ('parachute', 'chute')
    ]
    """
    def __init__(
        self,
        source_classname: str = 'tench',
        target_classname: str = 'cassette player',
        root: str = ROOT,
        dim: int = 30,
        reg: float = 0.01,
        distance: str | Callable = 'l2',
    ) -> None:
        """Initialize the parameters"""
        self.source_classname = source_classname
        self.target_classname = target_classname
        self.root = root
        self.dim = dim
        self.reg = reg
        self.distance = distance

        """Initialize the Resnet18 model"""
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.eval()
        # The transforms for the Resnet18 model is already defined in
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
        self.resnet18_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

        """Initialize the Imagenette dataset"""
        # transform: prepare the images for the Resnet18 model
        self.Imagenette = torchvision.datasets.Imagenette(root=root, split='train', size='full', download=True,
                                                          transform=self.resnet18_transforms)
        self._process_Imagenette()

        """Set the source/target distribution and the cost matrix"""
        self.source_distribution = self._get_distribution(source_classname)
        self.target_distribution = self._get_distribution(target_classname)
        self.cost_matrix = self._get_cost_matrix(source_classname, target_classname, distance)
        self.description = f"Imagenette.{source_classname}.{target_classname}.dim={dim}.norm={distance}.reg={reg}"

    def _get_classname(self, idx: int):
        """Get the classname: select the first name"""
        return self.Imagenette.classes[idx][0]

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
    
    def read_all_class(self):
        """Read the processed Imagenette dataset"""
        save_path = os.path.join(self.root, f'imagenette2-{self.dim}.pkl')
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    def read_processed_class(self, classname: str) -> np.ndarray:
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

    def _get_distribution(self, classname: str) -> np.ndarray:
        # vectors is an ndarray of shape (n, dim), where n is the number of images in the class
        vectors = self.read_processed_class(classname)
        n = vectors.shape[0]
        return 1.0 / n * np.ones(n)
    
    def _get_cost_matrix(self, source_classname: str, target_classname: str,
                        distance: str | Callable = 'l2') -> np.ndarray:
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

        return cost_matrix / np.max(cost_matrix)

class Synthetic1OT(BaseOT):

    def __init__(
        self,
        n: int = 100,
        m: int = 100,
        seed: int = 42,
    ) -> None:
        """Initialize the synthetic dataset"""
        self.n = n
        self.m = m
        self.rng = np.random.default_rng(seed)
        self.source_distribution = np.ones(n) / n
        self.target_distribution = np.ones(m) / m
        self.cost_matrix = self.rng.uniform(0, 1, (n, m))
        self.description = f"Synthetic1.n={n}.m={m}"
    
class Synthetic2OT(BaseOT):

    def __init__(
        self,
        n: int = 100,
        m: int = 100,
    ) -> None:
        x1 = 5 * np.arange(n) / (n - 1)
        x2 = 5 * np.arange(m) / (m - 1)
        source_density = lambda x: expon(scale=1.0).pdf(x)
        target_density = lambda x: 0.2 * norm(loc=1.0, scale=0.2).pdf(x) + 0.8 * norm(loc=3.0, scale=0.5).pdf(x)
        self.source_distribution = source_density(x1) / np.sum(source_density(x1))
        self.target_distribution = target_density(x2) / np.sum(target_density(x2))
        self.cost_matrix = np.square(x1.reshape(n, 1) - x2.reshape(1, m))
        self.cost_matrix = self.cost_matrix / np.max(self.cost_matrix)
        self.description = f"Synthetic2.n={n}.m={m}"