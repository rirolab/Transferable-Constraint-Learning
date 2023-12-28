# Inverse Constraint Learning and Generalization by Transferable Reward Decomposition

### [Project Page](https://sites.google.com/view/transferable-cl/) | [Paper](https://arxiv.org/abs/2306.12357) | [Video](https://www.youtube.com/watch?v=jpP_7XiR46c)


[Jaehwi Jang](),
[Minjae Song](),
[Daehyung Park](https://sites.google.com/site/daehyungpark), <br>
[RIROLAB](https://rirolab.kaist.ac.kr/), KAIST.

This is the official implementation of the paper "Inverse Constraint Learning and Generalization by Transferable Reward Decomposition".

[![ICL Video](https://rirolab.kaist.ac.kr/assets/research/2023_RAL_TCL_demo.gif)](https://www.youtube.com/watch?v=jpP_7XiR46c)


## Installation

### Preliminaries

This code implementation is based on [Imitation(#505)](https://github.com/HumanCompatibleAI/imitation/pull/505) and [Stable baselines3(1.6.0)](https://github.com/DLR-RM/stable-baselines3/tree/v1.6.0).

### Requirements

First, install Stable Baselines3 contrib with pip editable:

```
cd stable-baselines3-contrib-master
pip install -e .
cd ..
```

Second, install required packages

```
pip install hydra-core==1.0.0
pip install tensorboard
```

Move to imitation_contrib directory and run the codes.
```
cd imitation_contrib
```


## Quick start

### 2D wallfollowing
```
cd imitation_contrib/src
python3 python3 tcl_twodconstraint.py env=wallfollowing
```
## Repository under construction
Coming soon!