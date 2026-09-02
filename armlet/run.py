import os

import rich
from rich.panel import Panel
from rich.pretty import Pretty

import hydra
from omegaconf import DictConfig, OmegaConf

from armlet.utils.configs import ArmletConfiguration
from armlet.federation import run_federation
from armlet.audit import run_post_hoc_audit
from armlet.hpo import run_hpo_federation


OmegaConf.register_new_resolver("sanitize_override_dirname", lambda x: x.replace(os.path.sep, "_"))
OmegaConf.register_new_resolver("keep_last_str", lambda x: x.split(".")[-1])
OmegaConf.register_new_resolver("concat", lambda x, y: x+y)


@hydra.main(version_base=None, config_path="configs", config_name="federation")
def main(cfg : DictConfig) -> None:
    armlet_cfg = ArmletConfiguration(cfg)
    rich.print(Panel(Pretty(armlet_cfg, expand_all=True), title="Configuration", width=200))

    if cfg.armlet.mode == "federation":
        run_federation(armlet_cfg)
    elif cfg.armlet.mode == "hpo":
        run_hpo_federation(cfg)
    elif cfg.armlet.mode == "post_hoc_audit":
        run_post_hoc_audit(armlet_cfg)
    else:
        pass


if __name__ == "__main__":
    main()
