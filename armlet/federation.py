import os
import hydra
from omegaconf import OmegaConf

from fluke import FlukeENV, DDict

from armlet.data import data_pipeline
from armlet.audit.auditor import OnlineAuditor

def run_federation(cfg: DDict) -> None:

    data_splitter, additional_data = data_pipeline(cfg)

    FlukeENV().configure(cfg)

    # Automatically adjust some hyperparameters in cfg
    input_size = _infer_input_size(data_splitter.data_container.clients_tr[0])
    cfg.method.hyperparameters.model.input_size = input_size
    if data_splitter.data_container.num_classes <= 2:
        cfg.method.hyperparameters.model.num_classes = 1  
    else:
        cfg.method.hyperparameters.model.num_classes = data_splitter.data_container.num_classes

    _anonymize_and_save_cfg_dict(cfg.to_dict(), cfg.paths.output_dir)

    if cfg.exp.train:

        fl_algo = hydra.utils.instantiate(
            cfg.method,
            n_clients=cfg.protocol.n_clients,
            data_splitter=data_splitter,
            additional_data=additional_data,
            _convert_="all",
            _recursive_=False,
        )

        if "audit" in cfg.keys():
            auditor = OnlineAuditor(
                **cfg.audit,
                output_dir=cfg.paths.output_dir
            )
        else:
            auditor = None

        log_name = f"{fl_algo.__class__.__name__} [{fl_algo.id}]"
        log = hydra.utils.instantiate(
            cfg.logger,
            name=log_name,
            auditor=auditor,
            log_dir=cfg.paths.output_dir,
        )
        fl_algo.set_callbacks([log])
        FlukeENV().set_logger(log)

        evaluator = hydra.utils.instantiate(
            cfg.eval.exclude("locals", "post_fit", "pre_fit", "server"),
            n_classes=data_splitter.data_container.num_classes,
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

def _infer_input_size(train_loader):
    if hasattr(train_loader, "tensors"):
        return train_loader.tensors[0].shape[-1]

    dataset = getattr(train_loader, "dataset", None)
    if dataset is not None:
        input_size = getattr(dataset, "input_size", None)
        if input_size is not None:
            return input_size

        num_samples = getattr(dataset, "num_samples", None)
        if num_samples is not None:
            return num_samples

        if hasattr(dataset, "tensors"):
            return dataset.tensors[0].shape[-1]

    X, _ = next(iter(train_loader))
    return X.shape[-1]

def _anonymize_and_save_cfg_dict(cfg_dict, output_dir):

    cfg_dict["paths"] = {
        "root_dir": ".",
        "data_dir": "${paths.root_dir}/datasets",
        "output_dir": "${hydra:runtime.output_dir}",
    }

    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, 'w') as config_file:
        OmegaConf.save(config=cfg_dict, f=config_file.name)
