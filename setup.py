from setuptools import setup, find_packages
import versioneer

setup(
    name="flex-format",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
