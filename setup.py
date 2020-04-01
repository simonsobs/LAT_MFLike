from setuptools import setup, find_packages

setup(
    name="mflike",
    version="1.0",
    description="SO LAT multi-frequency likelihood for cobaya",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
        "cobaya @ git+https://github.com/CobayaSampler/cobaya@devel#egg=cobaya",
        "sacc>=0.3.0",
    ],
    package_data={"mflike": ["MFLike.yaml"]},
)
