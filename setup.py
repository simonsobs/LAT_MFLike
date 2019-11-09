from setuptools import setup, find_packages

setup(name="mflike",
      version='1.0',
      description='SO LAT multi-frequency likelihood for cobaya',
      zip_safe=True,
      packages = find_packages(),
      install_requires=['numpy', 'fgspectra', 'cobaya', 'sacc']
      )

