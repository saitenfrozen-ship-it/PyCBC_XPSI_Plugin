#!/usr/bin/env python
"""
Setup file for XPSI plugin model for PyCBC.
"""

from setuptools import Extension, setup, Command
from setuptools import find_packages

VERSION = '0.1.dev1'
NAME = 'pycbc_xpsi'

setup(
    name=NAME,
    version=VERSION,
    description='XPSI plugin model for PyCBC',
    author='Parth Sastry',
    author_email='psastry@umassd.edu',
    url='http://www.pycbc.org/',
    download_url='https://github.com/parthsastry/pycbc-xpsi/tarball/v%s' %
                 VERSION,
    keywords=['pycbc', 'xpsi', 'nicer', 'bayesian inference',
              'gravitational waves', 'x-ray astronomy',
              'multimessenger astronomy'],
    install_requires=['pycbc', 'xpsi'],
    packages=find_packages(),
    entry_points={
        "pycbc.inference.models": [
            "pycbc_xpsi_stu = pycbc_xpsi:XPSI_STUModel", 
            "pycbc_xpsi_pdtu = pycbc_xpsi:XPSI_PDTUModel",
        ],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
