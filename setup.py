import platform

from setuptools import find_packages, setup

install_requires = open("src/requirements.txt").read().splitlines()

if platform.system().lower() == "linux":
    install_requires.extend(["liger-kernel==0.4.2", "triton==3.0.0", "torch==2.4.0+cu118"])

setup(
    name="cot-icl-lab",
    version="0.1.0",
    author="Vignesh Kothapalli",
    author_email="k.vignesh1420@gmail.com",
    description="A library to train and evaluate transformers for tokenized chain-of-thought in-context learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kvignesh1420/cot-icl-lab",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    python_requires=">=3.10",
)
