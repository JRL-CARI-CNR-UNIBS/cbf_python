from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'cbf_python'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'params_csv'), glob('cbf_python/params_csv/*')),
        (os.path.join('share', package_name, 'skeleton_vectors'), glob('cbf_python/skeleton_vectors/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nyquist',
    maintainer_email='samuele.sandrini@polito.it',
    description='Modular Control Barrier Function (CBF) and Safe Trajectory Scaling for Manipulators in ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_cbf_optimal = cbf_python.examples.run_cbf_optimal:main',
            'run_dynamic_polynomial = cbf_python.examples.run_dynamic_polynomial:main',
            'run_gaussian_control = cbf_python.examples.run_gaussian_control:main',
            'run_obstructive_test = cbf_python.examples.run_obstructive_test:main',
            'run_cbf_pid = cbf_python.examples.run_cbf_pid:main',
            'run_dynamic_pol_subsequent = cbf_python.examples.run_dynamic_pol_subsequent:main',
            'run_optimization = cbf_python.examples.run_optimization:main',
            'run_optimization_poly = cbf_python.examples.run_optimization_poly:main',
            'run_optimization_gpr = cbf_python.examples.run_optimization_gpr:main',
            'run_optimization_obstructive = cbf_python.examples.run_optimization_obstructive:main',
            'plot_metrics = cbf_python.examples.plot_metrics:plot_comparison',
            'rebuild_dataset = cbf_python.examples.rebuild_dataset:main',
            'debug_cost = cbf_python.examples.debug_cost:main',
            'bridge_sine_test = cbf_python.examples.bridge_sine_test:main',
        ],
    },
)
