Maglab: Differentiable vector tomography based on micromagnetic model.

#### Reference:

Boyao Lyu, Shihua Zhao, Yibo Zhang, Weiwei Wang, Fengshan Zheng, Rafal E. Dunin-Borkowski, Jiadong Zang, Haifeng Du. (2024). *Three-dimensional magnetization reconstruction from electron optical phase images with physical constraints*. Science China-Physics Mechanics & Astronomy, 67(11), 117511. 

#### Update:

We have now determined that the Jacobian of the energy functional for the discretized spin array is equivalent to the effective field in micromagnetics. Consequently, the Adam optimizer used in our paper has now been replaced by a classic Barzilai-Borwein method, leading to faster convergence and eliminating the need for gradient propagation towards spherical coordinates.

For more details and usage of the driver, please refer to `maglab/sd.py` and `examples/target_skyrmion/main.ipynb`.

#### Installation:

1.Create a new conda environment:

```shell
conda create -n maglab python=3.9
conda activate maglab
```

2.Install [Pytorch](https://pytorch.org/get-started/locally/)

3.Install other requirements by:

```
pip install -r requirements.txt
```

4.Run setup.py:

```
python setup.py install
```

5.Check the demo in examples folder.