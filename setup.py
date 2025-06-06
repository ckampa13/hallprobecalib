from setuptools import setup, find_packages

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='hallprobecalib',
      version='0.1',
      description='Tools to analyze hall probe calibration data for Mu2e.',
      url='https://github.com/ckampa13/hallprobecalib',
      author='Cole Kampa',
      author_email='ckampa13@gmail.com',
      license_files = ('LICENSE',),
      python_requires='>=3.6',
      #packages=['hallprobecalib'],
      packages=find_packages(),
      install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'lmfit'],
      zip_safe=False)
