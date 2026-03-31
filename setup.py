from setuptools import find_packages, setup


setup(
    name="wicat",
    version="0.1.0",
    description="Anonymous WiCAT minimal package",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "xformers",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "einops",
        "omegaconf",
        "tqdm",
        "requests",
        "beautifulsoup4",
        "opencv-python",
        "pynwb",
        "pyyaml",
    ],
)
