# numerical-experiments

This is an improved version of the repository [SSNS](https://github.com/TangZihao1997/SSNS),
which is used to run the numerical experiments
in the paper [SSNS](https://openreview.net/forum?id=Nmmiyjw7Xg) and [SPLR](https://www.baidu.com/)

## Environment Preparation

Create a new environment `num`, install dependencies
```bash
conda create -n num python=3.13
conda activate num
conda install pip
pip install scikit-learn scipy tqdm numpy seaborn ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Installing the Latest Version of [regot-python](https://github.com/yixuan/regot-python)

```bash
conda activate num
git clone --depth=1 https://github.com/yixuan/regot-python.git 
cd regot-python
pip install . -r requirements.txt
```

## Running the Experiments

Running `main.ipynb` saves the results and plots in `save` folder.
The experiments include:
- ([Fashion-](https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html?highlight=fashion+mnist#torchvision.datasets.FashionMNIST))[MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html): Optimal transport between two images.
- [Imagenette](https://github.com/fastai/imagenette): Optimal transport between two classes of images.
- Synthetic Data: Optimal transport between two given distributions and cost matrices.