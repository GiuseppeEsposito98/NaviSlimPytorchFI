from setuptools import setup, find_packages
import pytorchfi


VERSION=1.0
DESCRIPTION ='pytorchfi description'
LONG_DESCRIPTION='pytorchfi description'


#seting up

setup(
    name="pytorchfi",
    version=VERSION,
    author="Giuseppe Esposito",
    author_email="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/GiuseppeEsposito98/NaviAPPFI",
    packages=find_packages(exclude=('tests', 'scripts', 'Demo')),
    python_requires='>=3.7',
    install_requires=[
        'torch==2.4.1',
        'torchvision>=0.13.2',
        'numpy',
        'pyyaml>=5.4.1',
        'scipy',
        'cython',
        'pycocotools>=2.0.2'
    ],
    extras_require={
        'test': ['pytest']
    },
    keywords=['pytorch','python'],
    classifiers=[
        "Development Status :: Apha",
        "Programming Language :: Python :: 3",
    ]
)
