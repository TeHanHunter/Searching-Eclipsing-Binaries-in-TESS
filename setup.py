import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="SEBIT",  # Replace with your own username
    version="0.0.1",
    author="Te Han",
    author_email="tehanhunter@gmail.com",
    description="Searching Eclipsing Binaries in TESS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TeHanHunter/Searching-Eclipsing-Binaries-in-TESS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "SEBIT"},
    packages=setuptools.find_packages(where="SEBIT"),
    python_requires=">=3.6",
)
