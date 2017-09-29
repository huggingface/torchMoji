from setuptools import setup

setup(
    name='torchmoji',
    version='1.0',
    packages=['torchmoji'],
    description='torchMoji',
    include_package_data=True,
    install_requires=[
        'emoji==0.4.5',
        'numpy==1.13.1',
        'scipy==0.19.1',
        'scikit-learn==0.19.0',
        'text-unidecode==1.0',
    ],
)
