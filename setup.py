from setuptools import setup, find_packages

setup(name="mflike",
      version="1.0",
      description="SO LAT multi-frequency likelihood for cobaya",
      zip_safe=True,
      packages = find_packages(),
      install_requires=[
          "fgspectra @ git+ssh://git@github.com/simonsobs/fgspectra@master#egg=fgspectra",
          "cobaya @ git+ssh://git@github.com/CobayaSampler/cobaya@external_modules#egg=cobaya",
          "sacc"
      ],
      package_data={"mflike": ["mflike.yaml"]},
)