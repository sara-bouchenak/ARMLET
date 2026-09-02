"""
This module contains the audit components for ``armlet``.
"""

from fluke import DDict

from armlet.audit.auditor import PostHocAuditor


def run_post_hoc_audit(cfg: DDict) -> None:
    auditor = PostHocAuditor(
        **cfg.audit,
        output_dir=cfg.paths.output_dir,
    )
    auditor.run_audit()
