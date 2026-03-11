<h1 align="center">
  <a href="https://pytorchfi.dev/"><img src="https://user-images.githubusercontent.com/7104017/75485879-22e79400-5971-11ea-9376-2d898034c23a.png" width="150"></a>
  <br/>
    PyTorchFI
  </br>
</h1>

[![codecov](https://codecov.io/gh/divadnauj-GB/pytorchfi_SC/branch/main/graph/badge.svg?token=WETJBPMAUN)](https://codecov.io/gh/divadnauj-GB/pytorchfi_SC)

# Introduction
This repo is a modified version of 
https://github.com/pytorchfi/pytorchfi.git
Adapted to run fault injections at the application level on the drone navigation model running on data extracted from Microsoft AirSim


## Background

PyTorchFI is a runtime perturbation tool for deep neural networks (DNNs), implemented for the popular PyTorch deep learning platform. PyTorchFI enables users to perform perturbation on weights or neurons of a DNN during runtime. It is extremely versatile for dependability and reliability research, with applications including resiliency analysis of classification networks, resiliency analysis of object detection networks, analysis of models robust to adversarial attacks, training resilient models, and for DNN interpertability.

An example of a use case for PyTorchFI is to simulate an error by performaing a fault-injection on an object recognition model.
|                                              Golden Output                                               |                                       Output with Fault Injection                                        |
| :------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
| ![](https://user-images.githubusercontent.com/7104017/85642872-7fb93980-b647-11ea-8717-8d16cb1c35b3.jpg) | ![](https://user-images.githubusercontent.com/7104017/85642867-7def7600-b647-11ea-89b9-570278c22101.jpg) |

## Contributors

Before contributing, please refer to our [contributing guidelines](https://github.com/pytorchfi/pytorchfi/blob/master/CONTRIBUTING.md).

- [Sarita V. Adve](http://sadve.cs.illinois.edu/) (UIUC)
- [Neeraj Aggarwal](https://neerajaggarwal.com) (UIUC)
- [Christopher W. Fletcher](http://cwfletcher.net/) (UIUC)
- [Siva Kumar Sastry Hari](https://research.nvidia.com/person/siva-hari) (NVIDIA)
- [Abdulrahman Mahmoud](http://amahmou2.web.engr.illinois.edu/) (UIUC)
- [Alex Nobbe](https://github.com/Alexn99) (UIUC)
- [Jose Rodrigo Sanchez Vicarte](https://jose-sv.github.io/) (UIUC)

## Citation

View the [published paper](http://rsim.cs.illinois.edu/Pubs/20-DSML-PyTorchFI.pdf). If you use or reference PyTorchFI, please cite:

```
@INPROCEEDINGS{PytorchFIMahmoudAggarwalDSML20,
author={A. {Mahmoud} and N. {Aggarwal} and A. {Nobbe} and J. R. S. {Vicarte} and S. V. {Adve} and C. W. {Fletcher} and I. {Frosio} and S. K. S. {Hari}},
booktitle={2020 50th Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W)},
title={PyTorchFI: A Runtime Perturbation Tool for DNNs},
year={2020},
pages={25-31},
}
```