from setuptools import setup, find_packages

setup(
    name='teccl',
    version='1.0.0',
    packages=find_packages(),
    # The topology DSL frontend is a git submodule, not an importable package (dashed name, no
    # __init__.py), so find_packages() cannot see it. Ship its files as data of the package that
    # loads them -- dsl_topology.py resolves them relative to its own __file__, so a copied
    # install has to carry them or every .topo file dies at load time.
    package_data={
        'teccl.topologies': [
            'topology-dsl-frontend/*.py',
            'topology-dsl-frontend/grammar.lark',
        ],
    },
    entry_points={
        'console_scripts': [
            'teccl = teccl.__main__:main',
        ],
    },
    install_requires=[
        'dataclasses',
        'argcomplete',
        'gurobipy',
        'numpy',
        'seaborn',
        'lark'
    ],
    python_requires='>=3.6',
)