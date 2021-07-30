from setuptools import setup

setup(
    name='ExtraGalacticFaraday',
    version='0.1',
    packages=['src', 'src.responses', 'src.model_library', 'src.model_library.galactic_models',
              'src.model_library.extra_galactic_models', 'src.helper_functions', 'src.helper_functions.data'],
    url='https://github.com/shutsch/ExtraGalacticFaraday',
    license='GPLv3',
    author='sebastian',
    author_email='hutsch@astro.ru.nl',
    description='A package to infer the extra-galactic RM-contributions',
    python_requires='>=3.6',
    classifiers=[
                                    "Development Status :: 5 - Production/Stable",
                                    "Topic :: Scientific/Engineering :: Mathematics",
                                    "Topic :: Scientific/Engineering :: Physics",
                                    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                                    "Operating System :: OS Independent",
                                    "Programming Language :: Python",
                                    "Intended Audience :: Science/Research"],

)
