<div align="center">

# DANO-PT: Domain-Agnostic Neural Operator for Particle Tracking

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.3-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/docs/2.3/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2.4-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Documentation

Links to the documentation for using the template and the main frameworks are provided in the header.

## Setup

Use the provided dockerfiles to build a suitable image. The image is not optimized, so ~45GB of storage space are required. See the associated <a href="https://github.com/Marcus1506/dano-pt/dockerfiles/README.md">
  README
</a> for more details.

## How to run

Train model with default configuration

```bash
python src/train.py
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=waterdrop
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
