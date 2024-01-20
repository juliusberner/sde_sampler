from setuptools import find_packages, setup

setup(
    name="sde_sampler",
    version="0.1.0",
    python_requires=">=3.9",
    zip_safe=True,
    packages=find_packages(include=["sde_sampler"]),
    author="Julius Berner",
    author_email="mail@jberner.info",
    description="Sampling via learned diffusions",
    install_requires=[
        # hydra
        "hydra-core==1.2.0",
        "hydra-joblib-launcher==1.2.0",
        "hydra-submitit-launcher==1.2.0",
        # torch additions
        "torchsde==0.2.6",
        "torchquad==0.4.0",
        "torch_ema==0.3",
        "pykeops==2.1.2",
        # logging, eval, and plotting
        "scipy==1.11.3",
        "pandas==2.1.3",
        "matplotlib==3.8.1",
        "wandb==0.16.0",
        "plotly==5.18.0",
        "kaleido==0.2.1",
    ],
    extras_require={
        "interactive": ["jupyter==1.0.0", "seaborn==0.13.0"],
        "dev": ["isort==5.10.1", "black==22.10.0"],
        "torch": ["torch", "torchvision"],
    },
)
