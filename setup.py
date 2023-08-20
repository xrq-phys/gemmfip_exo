from setuptools import find_packages, setup

setup(
    name='gemmfip_exo',
    version='0.0.1',
    description='Kernel generation templates for GEMMFIP',
    url='https://github.com/xrq-phys/gemmfip_exo/',
    author='RuQing G. Xu',
    packages=find_packages(exclude=['.github']),
    install_requires=[
        'exo_lang @ git+https://github.com/exo-lang/exo@master',
    ],
)
