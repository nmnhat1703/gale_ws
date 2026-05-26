from setuptools import setup, find_packages

setup(
    name='nmpc_s500',
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    py_modules=['nmpc_s500'],
    install_requires=[
        'numpy',
        'scipy',
        'pyyaml',
        'acados_template',
    ],
    author='nhat',
    author_email='nicktaonemay@gmail.com',
    description='ROS 1 NMPC controller for S500 quadrotor with MAVROS',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
