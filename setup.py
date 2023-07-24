from setuptools import find_packages, setup

import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="mflike",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simons Observatory Collaboration Power Spectrum Group aka so_ps 1 & 2",
    url="https://github.com/simonsobs/LAT_MFLike",
    description="SO LAT multi-frequency likelihood for cobaya",
    long_description=readme,
    long_description_content_type="text/x-rst",
    license="BSD license",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "fgspectra>=1.1.0",
        "syslibrary>=0.1.0",
        "cobaya>=3.3.0",
        "sacc>=0.9.0",
    ],
    package_data={"mflike": ["MFLike.yaml"]},
)
