import torchvision
import numpy as np
from typing import Callable

class OT:
    def __init__(self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray):
        """Initialize the OT problem"""
        self.source = source
        self.target = target
        self.cost_matrix = cost_matrix

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
         
    def get_cost_matrix(self, metric: str | Callable = 'l2', normailze: bool = True) -> np.ndarray:
        """Get the cost matrix"""
        if metric == 'l1' or 'cityblock':
            metric = lambda x, y: np.sum(np.abs(x - y))
        elif metric == 'l2' or 'euclidean':
            metric = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
        elif metric == 'l2_squared' or 'sqeuclidean':
            metric = lambda x, y: np.sum((x - y) ** 2)
        else:
            pass

        coordinates = np.array([[i, j] for i in range(MnistOT.n) for j in range(MnistOT.n)])
        cost_matrix = np.zeros((MnistOT.n_flattend, MnistOT.n_flattend))

        for i in range(MnistOT.n_flattend):
            for j in range(MnistOT.n_flattend):
                cost_matrix[i, j] = metric(coordinates[i], coordinates[j])
        
        return cost_matrix / np.max(cost_matrix) if normailze else cost_matrix
    
    def get_SinkhornOT(self, source_idx: int, target_idx: int,
                      metric: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_distribution = self.get_distribution(source_idx)
        target_distribution = self.get_distribution(target_idx)
        cost_matrix = self.get_cost_matrix(metric)
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
    
    def get_cost_matrix(self, metric: str | Callable = 'l2', normailze: bool = True):
        """Get the cost matrix"""
        if metric == 'l1' or 'cityblock':
            metric = lambda x, y: np.sum(np.abs(x - y))
        elif metric == 'l2' or 'euclidean':
            metric = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
        elif metric == 'l2_squared' or 'sqeuclidean':
            metric = lambda x, y: np.sum((x - y) ** 2)
        else:
            pass

        coordinates = np.array([[i, j] for i in range(FashionMnistOT.n) for j in range(FashionMnistOT.n)])
        cost_matrix = np.zeros((FashionMnistOT.n_flattend, FashionMnistOT.n_flattend))

        for i in range(FashionMnistOT.n_flattend):
            for j in range(FashionMnistOT.n_flattend):
                cost_matrix[i, j] = metric(coordinates[i], coordinates[j])
        
        return cost_matrix / np.max(cost_matrix) if normailze else cost_matrix
    
    def get_SinkhornOT(self, source_idx: int, target_idx: int,
                        metric: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_distribution = self.get_distribution(source_idx)
        target_distribution = self.get_distribution(target_idx)
        cost_matrix = self.get_cost_matrix(metric)
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
    
    def get_cost_matrix(self, metric: str | Callable = 'l2',
                        source_points: np.ndarray = None,
                        target_points: np.ndarray = None,
                        normalize: bool = True):
        """Get the cost matrix"""
        if metric == 'l1' or 'cityblock':
            metric = lambda x, y: np.abs(x - y)
        elif metric == 'l2' or 'euclidean':
            metric = lambda x, y: np.sqrt((x - y) ** 2)
        elif metric == 'l2_squared' or 'sqeuclidean':
            metric = lambda x, y: (x - y) ** 2
        elif metric == 'random':
            cost_matrix = self.rng.uniform(0, 1, (len(source_points), len(target_points)))
            return cost_matrix / np.max(cost_matrix) if normalize else cost_matrix

        cost_matrix = np.zeros((len(source_points), len(target_points)))
        for i in range(len(source_points)):
            for j in range(len(target_points)):
                cost_matrix[i, j] = metric(source_points[i], target_points[j])
        
        return cost_matrix / np.max(cost_matrix) if normalize else cost_matrix
    
    def get_SinkhornOT(self, source_density_fn: Callable, target_density_fn: Callable,
                       source_spacing: str = 'uniform', target_spacing: str = 'uniform',
                       source_low: float = 0.0, source_high: float = 5.0,
                       target_low: float = 0.0, target_high: float = 5.0,
                       metric: str | Callable = 'l2', reg: float = 0.01) -> SinkhornOT:
        source_points = self.get_points(self.n, source_spacing, source_low, source_high)
        target_points = self.get_points(self.m, target_spacing, target_low, target_high)
        source_distribution = self.get_distribution(source_density_fn, source_points)
        target_distribution = self.get_distribution(target_density_fn, target_points)
        cost_matrix = self.get_cost_matrix(metric, source_points, target_points)
        return SinkhornOT(source_distribution, target_distribution, cost_matrix, reg)
