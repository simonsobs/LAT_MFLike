from setuptools import find_packages, setup

import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="mflike",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="SO LAT multi-frequency likelihood for cobaya",
    long_description=readme,
    long_description_content_type="text/x-rst",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@act_sz_x_cib#egg=fgspectra",
        "syslibrary @ git+https://github.com/simonsobs/syslibrary@master#egg=syslibrary",
        "cobaya>=3.1.0",
        "sacc>=0.4.2",
    ],
    package_data={"mflike": ["MFLike.yaml"]},
)
