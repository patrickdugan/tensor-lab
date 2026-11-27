"""Offline tests for the exploratory Ramsey model-check companion."""
import math
import numpy as np
import ramsey_modelcheck as M


def test_constant_phase_matches_analytic_ramsey():
    theta=[0.04,0.12,-0.03,0.98,0.99]
    for state in (0,1):
        for m in (0,1,2,8):
            phase=theta[0]+m*(theta[1]+(-1)**state*theta[2]+state*math.pi)
            for axis in ("X","Y"):
                truth=theta[3]*theta[4]**m*(math.cos(phase) if axis=="X" else math.sin(phase))
                assert abs(M.probe_expectation(m,state,axis,theta)-truth)<1e-12
                assert abs(M.probe_expectation(m,state,axis,theta+[0],True)-truth)<1e-12


def test_relaxation_mixture_stays_physical():
    for m in (0,1,2,8,32):
        theta=[0.04,0.12,-0.03,0.98,0.99,0.1,0.03]
        x=M.probe_expectation(m,1,"X",theta,True,True)
        y=M.probe_expectation(m,1,"Y",theta,True,True)
        assert x*x+y*y<=1+1e-12


def test_binomial_deviance_endpoints_finite():
    for k in (0,1,50,99,100):
        assert np.isfinite(M.deviance(k,100,0.2))
        assert M.deviance(k,100,0.2)>=-1e-12
