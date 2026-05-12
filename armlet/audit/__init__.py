"""
This module contains the audit components for ``armlet``.
"""

import os
import pandas as pd

from armlet.audit.auditor import PostHocAuditor

from armlet.results_analysis.load_metrics import load_df_multirun
from armlet.results_analysis.utils import compute_metrics_name_dict, preprocess_data


def audit_model_from_results_json(exp_dir, cfg_paths, **kwargs):

    auditor = PostHocAuditor(**kwargs)

    exp_dir = os.path.join(cfg_paths.root_dir, "outputs", exp_dir)
    list_df = []
    for metrics_type in ["perf_global", "perf_locals", "perf_prefit", "perf_postfit"]:
        list_df.append(load_df_multirun(exp_dir, metrics_type))
    df = pd.concat(list_df, axis=0, ignore_index=True)

    metrics_by_cat, other_columns = compute_metrics_name_dict(df.columns.tolist())
    df = preprocess_data(df, metrics_by_cat, other_columns)

    auditor.audit_df_results(df, other_columns)
    #auditor.results.to_csv(os.path.join(cfg_paths.output_dir, "audit_results.csv"))
    #auditor.perc_success.to_csv(os.path.join(cfg_paths.output_dir, "audit_perc_success.csv"))

    print(auditor.results)
    print(auditor.perc_success)
