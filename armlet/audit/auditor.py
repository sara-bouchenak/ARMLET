import pandas as pd

from armlet.results_analysis.utils import filter_and_aggregate_metrics


class PostHocAuditor:

    def __init__(
            self,
            metrics_type_to_audit,
            last_n_rounds = None,
            agg_func = None,
            **kwargs,
        ) -> None:

        self.metrics_type_to_audit = metrics_type_to_audit

        self.properties = ["objective", "threshold", "agg_func", "last_n_rounds"]

        self.global_props = {
            "last_n_rounds": last_n_rounds,
            "agg_func": agg_func,
        }

        self.metrics_props_by_cat = init_metrics_props_by_cat(
            kwargs,
            self.global_props,
            self.properties,
        )

        self.results = None
        self.perc_success = None

    def audit_df_results(self, df, other_columns):

        metrics_type = [key for key, val in self.metrics_type_to_audit.items() if val]
        df = df[df["metrics_type"].isin(metrics_type)]

        info_columns = [col for col in other_columns["info"] if col != "round"]
        group_by_columns = other_columns["method_pars"] + info_columns

        report_list = []
        for cat, metrics_props in self.metrics_props_by_cat.items():

            for metric, metric_props in metrics_props.items():

                if metric in df.columns:

                    df_agg = filter_and_aggregate_metrics(
                        df[[metric] + other_columns["info"] + other_columns["method_pars"]],
                        metrics_by_cat={"": [metric]},
                        other_columns=other_columns,
                        last_n_rounds=metric_props["last_n_rounds"],
                        agg_func=metric_props["agg_func"],
                    )

                    df_results = df_agg.groupby(group_by_columns).apply(
                        compare_metric_values_with_threshold,
                        metric=metric,
                        metric_props=metric_props,
                    )

                else:
                    df_results = df.groupby(group_by_columns).apply(compute_df_result_with_error)

                df_results = df_results.reset_index(level=group_by_columns, drop=False)
                df_results["metric"] = metric
                df_results["category"] = cat
                for prop in self.properties:
                    df_results[prop] = metric_props[prop]

                report_list.append(df_results)

        results = pd.concat(report_list, axis=0)
        self.results = results.set_index(["category", "metric", "metrics_type", "source"])

        # Compute percentage of success by cat
        df_perc_success = results.groupby(["category", "metrics_type"]).apply(
            lambda df_lambda: (df_lambda["status"].value_counts()/len(df_lambda)).mul(100).round(2)
        )
        self.perc_success = df_perc_success.unstack(level=-1).fillna(0.0)

def compare_metric_values_with_threshold(df_group, metric, metric_props):

    metric_values = df_group[metric].values

    if metric_props["threshold"] is None:
        return "error", metric_values

    if metric_props["objective"] == "above":
        if (metric_values >= metric_props["threshold"]).all():
            result = "pass"
        else:
            result = "fail"
    elif metric_props["objective"] == "below":
        if (metric_values <= metric_props["threshold"]).all():
            result = "pass"
        else:
            result = "fail"
    else:
        result = "error"

    df_result = pd.DataFrame([[result, metric_values]], columns=["status", "value"])
    return df_result

def compute_df_result_with_error(df_group):
    df_result = pd.DataFrame([["error", []]], columns=["status", "value"])
    return df_result

def init_metrics_props_by_cat(other_cfg, global_props, properties):

        metrics_props_by_cat = {}

        for key, val in other_cfg.items():

            if "_metrics" in key:
                cat = key.split("_metrics")[0]
                metrics_props_by_cat[cat] = val

                for metric in metrics_props_by_cat[cat].keys():
                    for property in properties:
                        if property not in metrics_props_by_cat[cat][metric].keys():
                            if property in global_props.keys():
                                metrics_props_by_cat[cat][metric][property] = global_props[property]
                            else:
                                metrics_props_by_cat[cat][metric][property] = None

        return metrics_props_by_cat
