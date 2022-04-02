[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gwilding/voidshape/HEAD?labpath=void_exploration_2d)

# Exploring voids and their shapes using TDA


Check out the binder directly in the browser, or explore the notebook locally.

[Gudhi](http://gudhi.gforge.inria.fr/) (Geometry Understanding in Higher Dimensions) is used for computations. It is available through [Anaconda](https://anaconda.org/conda-forge/gudhi), and can be installed using

```
conda install -c conda-forge gudhi 
```

In addition to `Gudhi`, only `numpy`, `scipy` and `matplotlib` are needed, as well as `tqdm` (for progressbars). Alternatively to installing those manually, you can create your own environment within conda from the [`environment.yml`](https://github.com/gwilding/cosmicwebpersistence/blob/main/environment.yml) file in the repository:

```
conda env create -f environment.yml
conda activate cosmic-tda
```
