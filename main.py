from xgb import run_xgb
from rfr import run_rfr
from svr_rbf import run_svr_rbf
from svr_poly_2d import run_svr_poly_2d
from svr_poly_3d import run_svr_poly_3d
from baysian import run_baysian
from baysian_polynomial import run_baysian_poly

if __name__ == '__main__':
    #latency
    # run_xgb("latency")
    # run_rfr("latency")
    # run_svr_rbf("latency")
    # run_svr_poly_2d("latency")
    # run_svr_poly_3d("latency")
    run_baysian("latency")
    #run_baysian_poly("latency")

    #tps
    # run_xgb("tps")
    # run_rfr("tps")
    # run_svr_rbf("tps")
    # run_svr_poly_2d("tps")
    # run_svr_poly_3d("tps")
    # run_baysian("tps")
    # run_baysian_poly("tps")




