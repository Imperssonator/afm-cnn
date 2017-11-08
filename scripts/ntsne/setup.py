from setuptools import setup

setup(
    name='ntsne',
    author='Brian DeCost',
    author_email='bdecost@andrew.cmu.edu',
    version='0.1',
    description='A simple numpy wrapper around bh_tsne.',
    py_modules=['ntsne'],
    install_requires=[
        'numpy',
    ],
)

    
