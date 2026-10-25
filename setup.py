from setuptools import find_packages, setup

setup(name="constellation-sr", version="1.0.0", packages=find_packages(),
      python_requires=">=3.9", install_requires=["torch", "fastapi", "sqlalchemy", "numpy", "Pillow"])
