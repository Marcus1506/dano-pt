FROM torch_geometric_base:local AS torch_geometric

RUN pip install --no-cache-dir \
    docker-pycreds==0.4.0 \
    einops==0.8.0 \
    hydra-core==1.3.2 \
    kappamodules==0.1.99 \
    lightning==2.4.0 \
    lightning-utilities==0.11.8 \
    omegaconf==2.3.0 \
    pyparsing==3.2.0 \
    rootutils==1.0.7 \
    scipy==1.14.1 \
    tqdm==4.67.0 \
    wandb==0.18.6 \
    yarl==1.17.1 \
    h5pickle==0.4.2 \
    torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html \
    matplotlib

# JupyterLab
RUN pip install --no-cache-dir \
    ipykernel==6.29.5 \
    ipywidgets==8.1.5 \
    nbformat \
    nbclient

RUN python3 -m ipykernel install --sys-prefix \
    --name dano-pt \
    --display-name "Python (dano-pt CUDA12.6)"

RUN ln -sf /usr/bin/python3 /usr/bin/python
