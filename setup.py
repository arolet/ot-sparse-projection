from setuptools import setup

setup(name='ot-sparse-projection',
      version='0.1',
      description='Sparse optimal transport projections',
      author='Antoine Rolet',
      author_email='antoine.rolet@gmail.com',
      packages=['ot-sparse-projection'],
      package_dir={'': 'src'},
      install_requires=[
          'PyWavelets',
          'numpy',
          'scipy==1.1.0',
          'matplotlib',
          'pillow',
          'scikit-image'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
