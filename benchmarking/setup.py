from setuptools import setup, find_packages

setup(
    name='mlpro_bm_suite',
    version='0.1.0',
    description='MLPro Benchmarking Suite – Scenarios, Benchmark Tests, and Results',
    author='MLPro Team',
    author_email='mlpro@listen.fh-swf.de',
    license='Apache Software License 2.0',
    packages=find_packages(),  
    install_requires=[],
    
    extras_require={
        "full": [
            "mlpro[full] @ git+https://github.com/fhswf/mlpro.git@main"
        ],
    },

    zip_safe=False
)
