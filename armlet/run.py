import os
import hydra

from omegaconf import DictConfig, OmegaConf
from fluke import FlukeENV, DDict

from armlet.utils.configs import ArmletConfiguration
from armlet.data import data_pipeline

OmegaConf.register_new_resolver("sanitize_override_dirname", lambda x: x.replace(os.path.sep, "_"))
OmegaConf.register_new_resolver("keep_last_str", lambda x: x.split(".")[-1])
OmegaConf.register_new_resolver("concat", lambda x, y: x+y)


def run_federation(cfg: DDict) -> None:

    data_splitter, val_data = data_pipeline(cfg)

    FlukeENV().configure(cfg)

    # Automatically adjust some hyperparameters in cfg
    input_size = data_splitter.data_container.clients_tr[0].tensors[0].shape[-1]
    cfg.method.hyperparameters.model.input_size = input_size
    if data_splitter.data_container.num_classes <= 2:
        cfg.method.hyperparameters.model.num_classes = 1  
    else:
        cfg.method.hyperparameters.model.num_classes = data_splitter.data_container.num_classes

    # Save config file
    cfg_to_save = cfg.to_dict()
    cfg_to_save["paths"]["output_dir"] = "${hydra:runtime.output_dir}"
    if "json_log_dir" in cfg_to_save["logger"].keys():
        cfg_to_save["logger"]["json_log_dir"] = "${paths.output_dir}"
    config_path = os.path.join(cfg.paths.output_dir, "config.yaml")
    with open(config_path, 'w') as config_file:
        OmegaConf.save(config=cfg_to_save, f=config_file.name)

    if cfg.exp.train:

        fl_algo = hydra.utils.instantiate(
            cfg.method,
            n_clients=cfg.protocol.n_clients,
            data_splitter=data_splitter,
            val_data=val_data,
            _convert_="all",
            _recursive_=False,
        )

        log_name = f"{fl_algo.__class__.__name__} [{fl_algo.id}]"
        log = hydra.utils.instantiate(cfg.logger, name=log_name)
        log.init(**cfg, exp_id=fl_algo.id)
        fl_algo.set_callbacks([log])
        FlukeENV().set_logger(log)

        evaluator = hydra.utils.instantiate(
            cfg.eval.exclude("locals", "post_fit", "pre_fit", "server"), 
            n_classes=data_splitter.data_container.num_classes,
            sensitive_attributes=cfg.data.dataset.sensitive_attributes,
        )
        FlukeENV().set_evaluator(evaluator)

        try:
            fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
        except Exception as e:
            log.log(f"Error: {e}")
            FlukeENV().force_close()
            FlukeENV.clear()
            log.close()
            FlukeENV().close_cache()
            raise e

        log.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg : DictConfig) -> None:
    custom_cfg = ArmletConfiguration(cfg)

    if cfg.exp.mode == "federation":
        run_federation(custom_cfg)
    else:
        pass


if __name__ == "__main__":
    main()
