import re
from pathlib import Path

from setuptools import find_packages
from setuptools import setup



setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[path.stem for path in Path("src").glob("*.py")],
    include_package_data=True,
    zip_safe=False,
)
