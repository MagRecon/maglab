Maglab: Micromagnetics based reconstruction of three-dimensional magnetic structures.

#### Features:

• Leverages differentiable programming for micromagnetic effective field calculation, avoiding manually encoding of mathematical expressions.

• Differentiable magnetic phase shifts simulation.

• Applying Labonte's steepest descent method on tomography.

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