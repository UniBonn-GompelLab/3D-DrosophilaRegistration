import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="image_registration_tool",
    version="0.1.0",
    description="A graphical interface to annotate and place landmarks on images, built with the PySimpleGUI TKinter framework",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["image_annotation_gui", "registration_module"],
    include_package_data=True,
    python_requires='>3.7.1',
    install_requires=["numpy>=1.20.3", "pandas==1.3.3", "Pillow==9.0.1", "PySimpleGUI==4.57.0"],
)
