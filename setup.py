from setuptools import setup, find_packages
import versioneer

setup(
    name="flex-format",
    url="https://github.com/AWehrhahn/flex-format",
    author="Ansgar Wehrhahn",
    author_email="ansgar.wehrhahn@physics.uu.se",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
