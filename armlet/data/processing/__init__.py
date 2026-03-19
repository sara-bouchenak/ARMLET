"""
This module contains the data processing functions.
"""

import hydra

from armlet.data.processing.utils import apply_processing_to_data


def data_processing_pipeline(data, cfg_data):

    if "processing" in cfg_data.keys():

        for step_name, cfg_processing_step in cfg_data.processing.items():

            if cfg_processing_step._apply_directly_to_subdata_:
                function = hydra.utils.call(
                    cfg_processing_step.exclude("_apply_directly_to_subdata_"),
                    _partial_=True,
                )
                data = apply_processing_to_data(data, function)

            else:
                data = hydra.utils.call(
                    cfg_processing_step.exclude("_apply_directly_to_subdata_"),
                    data=data,
                )

    return data
