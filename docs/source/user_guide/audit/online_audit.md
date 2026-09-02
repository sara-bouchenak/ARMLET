(audit_run_online)=

# Online audit

Audit can be performed in an online fashion during the federation mode.
Its works similarly to the post-hoc audit mode, but it is repeated at the end of each round of the FL experiment with the accumulated metrics.

## How to run ?

To enable the online audit feature, users must add the `audit` config group, such as in the following example:
```bash
armlet +audit=online
```

```{eval-rst}

.. warning::
    Unlike the post-hoc audit mode, the ``exp_dir`` config value should not be specified in the configurations.

```

## Online audit outputs

At the end of each FL round, **ARMLET** will generate the following outputs:
- a dataframe presenting the results of each audit test performed during the round;
- a dataframe that summarizes the audit by aggregating the audit results (of the round) and providing sucess rates grouped by metric categories and types.

At the end of the FL process, **ARMLET** will save a dataframe containing the results of **all** audit tests.
