from setuptools import setup, find_packages

import unittest
import doctest

# Read in the version number
exec(open('src/min_var/version.py', 'r').read())

requirements = ["numpy"]

# with open('README.rst') as readme_file:
#     readme = readme_file.read()

def test_suite():
    """Discover all tests in the tests dir"""
    test_loader = unittest.TestLoader()
    # Read in unit tests
    test_suite = test_loader.discover('tests')

    # Read in doctests from README
    test_suite.addTests(doctest.DocFileSuite('README.md',
                                             optionflags=doctest.ELLIPSIS))
    return test_suite

setup(
    name = "minimal variance",
    version = __version__,
    install_requires= requirements,
    author = "Emily O'Riordan",
    author_email = "oriordane1@cardiff.ac.uk",
    packages=find_packages('src'),
    package_dir={"": "src"},    
    test_suite='setup.test_suite',
    license='The MIT License (MIT)',
    description = "A method of finding the 'minimal-variance' distance, to find distances in correlated degenerate datasets."
    )