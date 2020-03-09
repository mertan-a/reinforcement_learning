from setuptools import setup, find_packages

setup(
    # Metadata
    name='DRL-Homework-2',
    version=1.0,
    author='Tolga Ok & Nazim Kemal Ure',
    author_email='ure@itu.edu.com',
    url='',
    description='Homework-2 BLG 604E ',
    long_description="",
    license='MIT',

    # Package info
    packages=["blg604ehw2",],
    install_requires=[
          "gym[atari]==0.10.9",
          "IPython==6.5.0",
          "matplotlib==2.2.3",
          "box2d-py==2.3.6",
          "Pillow==5.2.0",
          "torch==0.4.1",
          "numpy==1.15.4",
          "ipywidgets==7.4.1",
          "jupyter"
      ],
    zip_safe=False
)