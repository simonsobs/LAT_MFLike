from setuptools import find_packages, setup

setup(
	name = 'plikmflike',
	version = '1.0',
	description = 'SO-Planck 2018 multi-frequency likelihood for cobaya.',
	zip_safe = True,
	packages = find_packages(),
	python_requires = ">=3.5",
	install_requires = [
		"cobaya>=3.1.0"
	],
	package_data = {
		"plikmflike" : [ "PlikMFLike.yaml" ]
	}
)
