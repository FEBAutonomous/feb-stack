import os
from glob import glob
from setuptools import setup

package_name = "feb_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "driver.yaml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Formula Electric at Berkeley",
    maintainer_email="pranavbhatttheonlyone@gmail.com",
    description="FEBAUTO Racing starter kit",
    license="BSD-2-Clause",
    entry_points={
        "console_scripts": [
            "reactive_driver = feb_driver.reactive_driver:main",
            "sysid_probe = feb_driver.sysid_probe:main",
        ],
    },
)
