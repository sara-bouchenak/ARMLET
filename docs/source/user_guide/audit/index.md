(audit)=

# Introduction

**ARMLET** allows for the auditing of FL models, whether using the post-hoc audit mode or directly integrated in the federation mode.
By defining Quality of Service (QoS) objectives and audit specification, users can produce audit reports that analyze the trade-offs of FL workloads across multiple QoS criteria, such as utility, fairness, privacy, and cost.

In the post-hoc audit mode, the audit is performed in a post-hoc fashion, by loading, processing, and analyzing the metrics recorded during one or multiple previous experiments. During the federation mode, online audit can be enabled to provide the same capabilities during training.

On the following subpages, we explain how to use the post-hoc audit mode, how to activate online audit, and detail all the related configurations.
