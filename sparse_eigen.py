import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=2, suppress=True)
np.random.seed(42)

SAVE = 'save'
EIGEN = 'eigen'

# set the plot configurations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'axes.titlepad': 10,
    'legend.fontsize': 15,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    # 'savefig.format': 'pdf', # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
})

def get_sparsification(n: int, m: int, num_zeros: int, repeat: int) -> list[list[tuple[int, int]]]:
    rows = np.arange(1, n)
    cols = np.arange(1, m - 1)
    coords = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
    sparsifications = []
    for _ in range(repeat):
        np.random.shuffle(coords)
        sparsifications.append([tuple(coord) for coord in coords[:num_zeros]])
    return sparsifications

def Hessian_eigenvalue(n, m, sparsifications: list[list[tuple[int, int]]], stride: int) \
    -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    results = []

    for sparsification in sparsifications:

        T = np.random.rand(n, m)
        H = np.zeros((n + m - 1, n + m - 1))
        H[:n, :n] = np.diag(T.sum(axis=1))
        H[n:, n:] = np.diag(T[:, :-1].sum(axis=0))
        H[:n, n:] = T[:, :-1]
        H[n:, :n] = T[:, :-1].T

        max_eigenvalues = []
        min_eigenvalues = []
        num_sparsifications = []

        for k, (i, j) in tqdm(enumerate(sparsification),
                              desc="Sparsification Progress"):

            if k % stride == 0:
                eigenvalues = np.linalg.eigvalsh(H)
                min_eigenvalues.append(eigenvalues[0])
                max_eigenvalues.append(eigenvalues[-1])
                num_sparsifications.append(k)

            H[i, n + j] = 0
            H[n + j, i] = 0

        results.append((
            np.array(min_eigenvalues),
            np.array(max_eigenvalues),
            np.array(num_sparsifications)
        ))
    
    return results

def main(n, m, num_zeros, stride, repeat):
    sparsifications = get_sparsification(n, m, num_zeros, repeat)
    results = Hessian_eigenvalue(n, m, sparsifications, stride)

    eigen_path = os.path.join(SAVE, EIGEN)
    if not os.path.exists(eigen_path):
        os.makedirs(eigen_path)
    
    title = f'Hessian Eigenvalues (n={n}, m={m})'
    plt.figure(figsize=(10, 6), dpi=300)
    plt.title(title)
    plt.xlabel('Sparsification Step')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    for i, result in enumerate(results):
        min_eigenvalues, max_eigenvalues, num_sparsifications = result
        if i == 0:
            plt.plot(
                num_sparsifications,
                min_eigenvalues,
                label='Min Eigenvalue',
                color='blue',
                linestyle='dashed',
                linewidth=0.8,
                alpha=0.4,
            )
            plt.plot(
                num_sparsifications,
                max_eigenvalues,
                label='Max Eigenvalue',
                color='red',
                linestyle='solid',
                linewidth=0.8,
                alpha=0.4,
            )
        else :
            plt.plot(
                num_sparsifications,
                min_eigenvalues,
                color='blue',
                linestyle='dashed',
                linewidth=0.8,
                alpha=0.4,
            )
            plt.plot(
                num_sparsifications,
                max_eigenvalues,
                color='red',
                linestyle='solid',
                linewidth=0.8,
                alpha=0.4,
            )
    plt.legend(loc='upper right')
    savefig_path = os.path.join(eigen_path, f'eigenvalues_n{n}_m{m}.png')
    plt.tight_layout()
    plt.savefig(savefig_path)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test how the eigenvalues change after sparsification')
    parser.add_argument('--n', type=int, default=100, 
                        help='Number of rows (default: 100)')
    parser.add_argument('--m', type=int, default=100,
                        help='Number of columns (default: 100)')
    parser.add_argument('--num-zeros', type=int, default=None,
                        help='Number of zeros to introduce (default: None, set [1:, 1:] to zero)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for eigenvalue calculation (default: 1)')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Number of times to repeat the sparsification (default: 10)')
    args = parser.parse_args()

    n, m = args.n, args.m
    num_zeros = min((n-1)*(m-2), args.num_zeros) \
                if args.num_zeros is not None \
                else (n-1) * (m-2)
    stride = args.stride
    repeat = args.repeat

    main(n, m, num_zeros, stride, repeat)