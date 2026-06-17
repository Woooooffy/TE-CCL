import os

import gurobipy as gp

_env = None


def get_gurobi_env() -> gp.Env:
    """
        Returns a process-wide Gurobi Env, creating it on first use.
        If GUROBI_WLSACCESSID/GUROBI_WLSSECRET/GUROBI_LICENSEID are set in the
        environment (e.g. on a remote machine using a WLS license), the env is
        created with those credentials. Otherwise falls back to the default
        local license behavior.
    """
    global _env
    if _env is None:
        if "GUROBI_WLSACCESSID" in os.environ:
            params = {
                "WLSACCESSID": os.environ["GUROBI_WLSACCESSID"],
                "WLSSECRET": os.environ["GUROBI_WLSSECRET"],
                "LICENSEID": int(os.environ["GUROBI_LICENSEID"]),
            }
            _env = gp.Env(params=params)
        else:
            _env = gp.Env()
    return _env
