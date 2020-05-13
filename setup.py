from setuptools import find_packages, setup

setup(
    name="mflike",
    version="1.0",
    description="SO LAT multi-frequency likelihood for cobaya",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
        "cobaya>=3.0",
        "sacc @ git+https://github.com/simonsobs/sacc@mflike_current#egg=sacc",
    ],
    package_data={"mflike": ["MFLike.yaml"]},
)
