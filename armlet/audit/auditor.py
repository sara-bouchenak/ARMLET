from abc import ABC, abstractmethod
import os
import pandas as pd

from armlet.audit.load_metrics import load_df_multirun, preprocess_df_metrics
from armlet.audit.load_metrics import filter_and_aggregate_metrics


class Auditor(ABC):

    def __init__(
        self,
        output_dir,
        metric_types_to_audit,
        last_n_rounds=None,
        agg_func=None,
        **kwargs,
    ) -> None:

        self.output_dir = output_dir

        self.metric_types = [key for key, val in metric_types_to_audit.items() if val]

        self.properties = [
            "objective",
            "threshold_medium",
            "threshold_good",
            "agg_func",
            "last_n_rounds",
        ]

        self.global_props = {
            "last_n_rounds": last_n_rounds,
            "agg_func": agg_func,
        }

        self.metrics_props_by_cat = _init_metrics_props_by_cat(
            kwargs,
            self.global_props,
            self.properties,
        )

        self.results = None
        self.perc_success = None

    @abstractmethod
    def run_audit(self):
        raise NotImplementedError

    def _audit_df_metrics(self, df) -> tuple[pd.DataFrame, pd.DataFrame]:

        group_by_columns = [col for col in df.columns if col not in ["round", "metric_value"]]
        group_by_columns_2 = [col for col in df.columns if col not in
                                  ["round", "metric_value", "metric_name", "metric_cat"]]

        report_list = []

        for cat, metrics_props in self.metrics_props_by_cat.items():

            for metric, metric_props in metrics_props.items():

                if metric in df["metric_name"].unique():

                    df_metric = df[df["metric_name"] == metric]

                    df_agg = filter_and_aggregate_metrics(
                        df_metric,
                        group_by_columns=group_by_columns,
                        last_n_rounds=metric_props["last_n_rounds"],
                        agg_func=metric_props["agg_func"],
                    )

                    df_results = df_agg.groupby(group_by_columns).apply(
                        _compare_metric_values_with_threshold,
                        metric_props=metric_props,
                    )
                    df_results = df_results.reset_index(level=group_by_columns, drop=False)

                else:

                    df_results = df.groupby(group_by_columns_2).apply(_compute_df_result_with_error)
                    df_results = df_results.reset_index(level=group_by_columns_2, drop=False)
                    df_results["metric_name"] = metric

                df_results["metric_cat"] = cat
                for prop in self.properties:
                    df_results[prop] = metric_props[prop]

                report_list.append(df_results)

        df_audit_results = pd.concat(report_list, axis=0)

        # Compute percentage of success by cat
        df_perc_success = df_audit_results.groupby(["metric_cat", "metric_type"]).apply(
            lambda df_lambda: (
                (df_lambda["audit_status"].value_counts() / len(df_lambda)).mul(100).round(2)
            )
        )
        df_perc_success = df_perc_success.unstack(level=-1).fillna(0.0)

        return df_audit_results, df_perc_success


def _init_metrics_props_by_cat(other_cfg, global_props, properties):

    metrics_props_by_cat = {}

    for key, val in other_cfg.items():
        if "_metrics" in key:
            cat = key.split("_metrics")[0]
            metrics_props_by_cat[cat] = val

            for metric in metrics_props_by_cat[cat].keys():
                for property in properties:
                    if property not in metrics_props_by_cat[cat][metric].keys():
                        if property in global_props.keys():
                            metrics_props_by_cat[cat][metric][property] = global_props[
                                property
                            ]
                        else:
                            metrics_props_by_cat[cat][metric][property] = None

    return metrics_props_by_cat


def _compare_metric_values_with_threshold(df_group, metric_props):

    metric_values = df_group["metric_value"].values

    if metric_props["threshold_medium"] is None:
        result = "error"

    elif metric_props["threshold_good"] is None:
        result = "error"

    elif metric_props["objective"] == "above":
        if (metric_values >= metric_props["threshold_good"]).all():
            result = "good"
        elif (metric_values >= metric_props["threshold_medium"]).all():
            result = "medium"
        else:
            result = "bad"

    elif metric_props["objective"] == "below":
        if (metric_values <= metric_props["threshold_good"]).all():
            result = "good"
        elif (metric_values <= metric_props["threshold_medium"]).all():
            result = "medium"
        else:
            result = "bad"

    else:
        result = "error"

    df_result = pd.DataFrame([[result, metric_values]], columns=["audit_status", "metric_values"])
    return df_result


def _compute_df_result_with_error(df_group):
    df_result = pd.DataFrame([["error", []]], columns=["audit_status", "metric_values"])
    return df_result


class PostHocAuditor(Auditor):

    def __init__(
        self,
        exp_dir,
        output_dir,
        metric_types_to_audit,
        last_n_rounds=None,
        agg_func=None,
        **kwargs,
    ) -> None:

        super().__init__(output_dir, metric_types_to_audit, last_n_rounds, agg_func, **kwargs)

        df_metrics = load_df_multirun(exp_dir, self.metric_types)
        df_metrics = preprocess_df_metrics(df_metrics, no_metric_cat=True)
        self.df_metrics = df_metrics[df_metrics["metric_type"].isin(self.metric_types)]

    def run_audit(self):
        df_audit_results, df_perc_success = self._audit_df_metrics(self.df_metrics)

        df_audit_results = df_audit_results.set_index(
            ["metric_cat", "metric_name", "metric_type", "source"]
        )
        print(df_audit_results)
        df_audit_results.to_csv(os.path.join(self.output_dir, "audit_results.csv"))

        print(df_perc_success)
        df_perc_success.to_csv(os.path.join(self.output_dir, "audit_perc_success.csv"))


class OnlineAuditor(Auditor):

    def __init__(
        self,
        output_dir,
        metric_types_to_audit,
        last_n_rounds=None,
        agg_func=None,
        **kwargs,
    ) -> None:

        super().__init__(output_dir, metric_types_to_audit, last_n_rounds, agg_func, **kwargs)

        self.metrics_list = []
        self.all_audit_results = pd.DataFrame()

    def add_metrics(
        self,
        metrics_dict,
        metric_type,
        round,
    ):

        if metric_type == "perf_global":
            source = "server"
            for metric_name, metric_value in metrics_dict.items():
                metric_row = [metric_type, metric_name, metric_value, source, round]
                self.metrics_list.append(metric_row)
    
        else:
            for source, metrics_subdict in metrics_dict.items():
                source = "client_{}".format(source) if source != "server" else source
                for metric_name, metric_value in metrics_subdict.items():
                    metric_row = [metric_type, metric_name, metric_value, source, round]
                    self.metrics_list.append(metric_row)

    def run_audit(self):
        columns = ["metric_type", "metric_name", "metric_value", "source", "round"]
        df_metrics = pd.DataFrame(self.metrics_list, columns=columns)
        df_metrics = preprocess_df_metrics(df_metrics, no_metric_cat=True)

        df_audit_results, df_perc_success  = self._audit_df_metrics(df_metrics)

        df_audit_results["round"] = df_metrics["round"].max()
        self.all_audit_results = pd.concat([self.all_audit_results, df_audit_results])
        self.all_audit_results = self.all_audit_results.reset_index(drop=True)

        df_audit_results = df_audit_results.set_index(
            ["round", "metric_cat", "metric_name", "metric_type", "source"]
        )
        print(df_audit_results)
        print(df_perc_success)

    def save_audit_results(self):
        self.all_audit_results = self.all_audit_results.set_index(
            ["round", "metric_cat", "metric_name", "metric_type", "source"]
        )
        self.all_audit_results.to_csv(os.path.join(self.output_dir, "audit_results.csv"))
