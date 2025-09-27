# Application of the R.McKenna mechanism to the adult dataset

This repository is based on the mechanism used by
[Ryan McKenna](https://www.ryanhmckenna.com) who won the
[first place](https://www.nist.gov/ctl/pscr/team-rmckenna) in the
[Differential Privacy Synthetic Data Challenge][nist-challenge] of the NIST
in 2018. It is a modification of [his submitted solution][submitted-solution]
to run on the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult).

[nist-challenge]: https://www.nist.gov/ctl/pscr/open-innovation-prize-challenges/past-prize-challenges/2018-differential-privacy-synthetic

[submitted-solution]: https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools/de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/rmckenna

## Installation

```shell
# Clone the R.McKenna repository as a submodule
git submodule update --init

# Install the Python dependencies in a virtual environment
uv sync
```

## Downloading the dataset

The adult dataset can be downloaded from
[this link](https://archive.ics.uci.edu/ml/datasets/adult). Afterward,
format it using `notebooks/adult-preprocess.ipynb` and generate the required
domain information file using `notebooks/adult-domain.ipynb`.

## Generating a synthetic dataset

Use the following command to generate a synthetic dataset. You can also
configure the parameters (use `--help` to list them).

```shell
python adult.py  # --help displays the parameters
```

## Simple test of the mbi module

```shell
uv run mbi/examples/toy_example.py
```

## GPU support

Check that the driver of your graphics card is installed and that it supports
cuda. Install cuda from
[the website of Nvidia](https://developer.nvidia.com/cuda-downloads) and reboot
your computer after the installation.

Check your version of cuda using `/usr/local/cuda/bin/nvcc --version` or
`nvcc --version`. If your version of cuda is 11, install the corresponding
version of pytorch using:

```shell
# If your version of cuda is >= 11
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Check that torch has access to cuda:

```shell
python -c "import torch; print(torch.cuda.is_available())"
```

A little modification has to be done on the sources of the `private-pgm`
repository. Add the following lines before the line 267 that sets the `diff`
variable:

```python
# If we are using the torch backend, the Q linear operator (of
# type matrix.Identity) was not a Tensor and generated an
# error due to this unaccepted format for Tensor operations.
if all((self.backend == 'torch',
        str(type(Q)) == "<class 'matrix.Identity'>")):
    import torch

    # First, we transform Q into a numpy array
    q_as_numpy_array = Q * np.identity(Q.shape[1])

    # Then, we format this numpy array to a Tensor
    Q = torch.as_tensor(q_as_numpy_array, dtype=torch.float32,
                        device=self.Factor.device)

    # Just a verification that the formatting of Q keeps the
    # same values. It is the case for the multiple executions
    # that I launched, you can keep or remove this as you want.
    assert np.array_equal(q_as_numpy_array, Q.cpu().numpy())
```

### Usage

You can generate a synthetic dataset using GPU by setting the backend parameter
to `torch`.

```shell
python adult.py --backend torch  # use --help instead to display the parameters
```

You can monitor the usage of the GPU by `watch -d -n 0.5 nvidia-smi`. You can
also use `nvtop`:

```shell
sudo apt install -y nvtop
nvtop
```
