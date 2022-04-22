from pathlib import Path
from typing import Union, Mapping, Any

import yaml


class ConfigReader:
    """
    Config parameters handler

    Parameters
    ----------
    config : [str, Path]
        config filepath in .yml format
    """

    @classmethod
    def load(cls, config_filepath: Union[str, Path]):
        """
        Class method to initialise config reader from path to config
        Parameters
        ----------
        config_filepath : [str, Path]
            config filepath in .yml format

        Returns
        -------

        """
        with open(config_filepath, "r") as stream:
            config = yaml.safe_load(stream)
        
        return cls(config)

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        
        for k, v in self.config.items():
            setattr(self, k, v)