from xgboost import run_xgb
from rfr import run_rfr
from svr_rbf import run_svr_rbf
from baysian import run_baysian
if __name__ == '__main__':
    #latency
    run_xgb("latency")
    run_rfr("latency")
    run_svr_rbf("latency")
    run_baysian("latency")

    #tps
    run_xgb("tps")
    run_rfr("tps")
    run_svr_rbf("tps")
    run_baysian("tps")



