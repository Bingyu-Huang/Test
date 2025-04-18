#!/usr/bin/env python

from setuptools import find_packages, setup

import os
import subprocess
import time

version = '0.1.0'

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha

sha = get_git_hash()
short_sha = sha[:7]

setup(
    name='eventdeblur',
    version=version,
    description='Event-Based Motion Deblurring with Diffusion Models',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='bingyu.huang@ugent.be',
    keywords='computer vision, deblurring, event camera, diffusion models',
    url='https://github.com/Bingyu-Huang/EventDeblur',
    packages=find_packages(exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    license='Apache License 2.0',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'torch>=1.7',
        'torchvision',
        'numpy',
        'opencv-python',
        'scikit-image'
    ],
    zip_safe=False)
