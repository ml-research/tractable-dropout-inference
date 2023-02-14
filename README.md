# Probabilistic Circuits That Know What They Don't Know

Code for our paper "Probabilistic Circuits That Know What They Don't Know" (https://arxiv.org/abs/2302.06544).

### Abstract

Probabilistic circuits (PCs) are models that allow exact and tractable probabilistic inference. In contrast to neural networks, they are often assumed to be well-calibrated and robust to out-of-distribution (OOD) data. In this paper, we show that PCs are in fact not robust to OOD data, i.e., they don't know what they don't know. We then show how this challenge can be overcome by model uncertainty quantification. To this end, we propose tractable dropout inference (TDI), an inference procedure to estimate uncertainty by deriving an analytical solution to Monte Carlo dropout (MCD) through variance propagation. Unlike MCD in neural networks, which comes at the cost of multiple network evaluations, TDI provides tractable sampling-free uncertainty estimates in a single forward pass. TDI improves the robustness of PCs to distribution shift and OOD data, demonstrated through a series of experiments evaluating the classification confidence and uncertainty estimates on real-world data.


## Citing This Work

To cite this work, refer to the following citation (in Bibtex format):

```
@misc{https://doi.org/10.48550/arxiv.2302.06544,
  doi = {10.48550/ARXIV.2302.06544},
  url = {https://arxiv.org/abs/2302.06544},
  author = {Ventola, Fabrizio and Braun, Steven and Yu, Zhongjie and Mundt, Martin and Kersting, Kristian}, 
  title = {Probabilistic Circuits That Know What They Don't Know},
  publisher = {arXiv},
  year = {2023},
}
```