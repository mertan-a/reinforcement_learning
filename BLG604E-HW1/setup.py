from setuptools import setup, find_packages

setup(
    # Metadata
    name='DRL-Homework-1',
    version=1.0,
    author='Tolga Ok & Nazim Kemal Ure',
    author_email='ure@itu.edu.com',
    url='',
    description='Homework-1 BLG 604E ',
    long_description="",
    license='MIT',

    # Package info
    packages=["blg604ehw1",],
    install_requires=[
          "gym==0.10.9",
          "IPython==6.5.0",
          "matplotlib==2.2.3",
          "numpy==1.16.1",
          "six==1.11.0",
          "jupyter"
      ],
    zip_safe=False
)