from setuptools import setup

setup(
    name='virtual_dark',
    version='',
    packages=['virtual_dark'],
    url='',
    license='',
    author='Jonas Nikula',
    author_email='',
    description='Package for somewhat-automated processing of film negatives',
    install_requires=['Pillow~=7.0.0',
                      'numpy~=1.18.2',
                      'opencv-contrib-python~=4.2.0.32',
                      'matplotlib~=3.2.1',
                      'imutils~=0.5.3',
                      'rawpy~=0.13.1']
)
