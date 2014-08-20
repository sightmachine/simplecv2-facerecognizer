from setuptools import setup, find_packages

import simplecv_facerecognizer


setup(name="simplecv2-facerecognizer",
      version=simplecv_facerecognizer.__version__,
      description="simplecv plugin that provides face recognition",
      long_description=("Plugin for simplecv library, framework for computer (machine) vision in Python, "
                        "providing a unified, pythonic interface to image acquisition, conversion, "
                        "manipulation, and feature extraction."),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Manufacturing',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
          'Topic :: Multimedia :: Graphics :: Graphics Conversion',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      keywords='opencv, cv, machine vision, computer vision, image recognition, simplecv',
      author='Sight Machine Inc',
      author_email='support@sightmachine.com',
      url='http://simplecv.org',
      license='BSD',
      packages=find_packages(exclude=['ez_setup']),
      zip_safe=False,
      install_requires=['simplecv>=2.0'],
      package_data={
          'simplecv_facerecognizer':
              ['data/haar.txt',
               'data/HaarCascades/*.xml',
               'data/test/standard/*.png',
               'data/test/standard/*.csv',
               'data/sampleimages/*.jpg',
               'data/FaceRecognizer/*.xml'
              ]
      },
      entry_points={
          'simplecv.image': [
              'simplecv_facerecognizer = simplecv_facerecognizer.image_plugin'
          ],
          'simplecv.factory': [
              'HaarFeature = simplecv_facerecognizer.features:HaarFeature'
          ]
      },
      )
