from setuptools import find_packages, setup

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()


with open("requirements.txt", "r") as req:
    REQUIREMENTS = req.read().splitlines()

setup(
    name="Deep Scattering Models",
    version="0.1.0",
    description=(
        "Physics of Electromagnetic Scattering Models explored " 
        "with Deep Learning tools."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=REQUIREMENTS,
    author="Julián Villa",
    author_email="jvilla@iafe.uba.ar",
    url="https://github.com/jotavecorta/DeepScatteringModels",
    packages=find_packages(),
    license="The MIT License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
    ],
)