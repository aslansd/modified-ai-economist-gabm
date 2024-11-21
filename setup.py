import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="modified-ai-economist-gabm",
    version="1.1.0",
    author="Aslan Satary Dizaji",
    author_email="asataryd@umich.edu",
    description="Incentives to Build Houses, Trade Houses, or Trade House Building Skills in Simulated Worlds under Various Governing Systems or Institutions: Comparing Multi-agent Reinforcement Learning to Generative Agent-based Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aslansd/modified-ai-economist-gabm",
    packages=setuptools.find_packages(),
    package_data={
        "modified_ai_economist_gabm": [
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)