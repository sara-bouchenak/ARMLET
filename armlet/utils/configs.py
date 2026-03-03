from omegaconf import DictConfig, OmegaConf

from fluke import DDict
from fluke.config import Configuration


class ArmletConfiguration(Configuration):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_ddict = DDict(cfg_dict)
        self.update(cfg_ddict)
