from setuptools import setup, find_packages

setup(
    name='edgeml',
    version='0.1.2',
    packages=find_packages(),
    description='library to enable distributed edge ml training and inference',
    url='https://github.com/youliangtan/edgeml',
    packages=find_packages(),
    author='auth',
    author_email='tan_you_liang@hotmail.com',
    license='MIT',
    install_requires=[
        'zmq',
        'typing',
        'typing_extensions',
        'opencv-python',
        'lz4',
    ],
    zip_safe=False
)
