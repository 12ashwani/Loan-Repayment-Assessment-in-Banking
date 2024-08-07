from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'




def get_reqirements(file_path)->List[str]:
    ''' This function will eturn the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(

    name='new_ml_project',
    version='0.0.1',
    author='Ashwani kumar',
    author_email='kumarashwani1208@gmail.com',
    packages=find_packages(),
    install_requires=get_reqirements('requirements.txt')
)