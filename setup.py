from setuptools import setup, find_packages

setup(
    name='deep_info_max_bottleneck',
    version='0.1.0',
    author='Peyman Poozesh',
    author_email='Poozesh.peyman@gmail.com',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_package',
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.1',
        'numpy>=1.19.2',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'deep_info_max_bottleneck=deep_info_max_bottleneck.cli:main',
        ],
    },
)