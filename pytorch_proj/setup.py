from setuptools import setup

setup(
    name='my_tf_proj', #project name
    version='0.1.0',
    description='my research library for deep learning',
    #url
    author='Brando Miranda',
    author_email='brando90@mit.edu',
    license='MIT',
    packages=['pytorch_pkg']
    install_requires=['numpy','maps','pdb','scikit-learn','scipy','sklearn','sympy','matplotlib','tk']
)
