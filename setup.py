"""Install CAE-T5."""
import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'caet5')
sys.path.append(version_path)
from version import __version__

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='caet5',
    version=__version__,
    description='Canditional auto-encoder text-to-text transfer transformer',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Leo Laugier, Ioannis Pavlopoulos, Jeffrey Sorensen, Lucas Dixon',
    author_email='leo.laugier@telecom-paris.fr',
    url='https://github.com/LeoLaugier/self_supervised_t5_style_transfer', # TODO
    # license='Apache 2.0', # TODO
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[ #TODO Update CAET5 to fit last versions of transformers and t5
        #"tensorflow",
        #"tensorflow-text",
        #"transformers==2.8.0",
        #"tfds-nightly",
        "google-api-python-client",
        "google-cloud_storage",
        "tensorflow_hub",
        "torchtext",
        "transformers==2.8.0",
        "t5==0.5.0",
    ],
    entry_points={
        'console_scripts': [
            'caet5 = caet5.main:console_entry_point'
        ],
    },
    classifiers=[
        # 'Development Status :: 4 - Beta', # TODO
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        # 'License :: OSI Approved :: Apache Software License', # TODO
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='text nlp nlg machinelearning deeplearning',
)