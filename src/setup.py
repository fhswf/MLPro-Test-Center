from setuptools import setup


setup(name='mlpro-best-practices',
version='2.0.1',
description='MLPro Best Practice Collection – Executable HowTos, Examples and Benchmarks',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro-best-practices'],

# Package dependencies for full installation
extras_require={
    "full": [
        "git+https://github.com/fhswf/mlpro.git@main#egg=mlpro[full]"
    ],
},

zip_safe=False)