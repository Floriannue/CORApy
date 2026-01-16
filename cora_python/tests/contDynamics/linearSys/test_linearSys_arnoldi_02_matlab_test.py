"""
test_linearSys_arnoldi_02_matlab_test - Translation of MATLAB test_Krylov_Arnoldi.m

This is a direct translation of the MATLAB unit test test_Krylov_Arnoldi.m
which checks the Arnoldi iteration against a fragment of the MATLAB 
implementation of expokit.

Source: cora_matlab/unitTests/contDynamics/linearSys/test_Krylov_Arnoldi.m
Translated: 2025-01-XX

Authors:       Translation of MATLAB test by Matthias Althoff
Written:       13-November-2018 (original MATLAB)
               2025-01-XX (Python translation)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.g.functions.helper.dynamics.contDynamics.linearSys.arnoldi import arnoldi


def aux_arnoldi_expokit(A, v, KrylovOrder):
    """
    Auxiliary function: Arnoldi iteration from expokit
    Direct translation of aux_Arnoldi_expokit from test_Krylov_Arnoldi.m
    """
    n = len(A)
    m = KrylovOrder
    
    # init V, H
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 2, m + 2))
    
    # norm of input vector
    normv = np.linalg.norm(v)
    beta = normv
    
    # init w (flatten to 1D if needed)
    w = v.flatten() if v.ndim > 1 else v
    
    # Arnoldi
    V[:, 0] = (1 / beta) * w
    for j in range(m):
        p = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = V[:, i].T @ p
            p = p - H[i, j] * V[:, i]
        s = np.linalg.norm(p)
        
        H[j + 1, j] = s
        if s > 0:
            V[:, j + 1] = (1 / s) * p
    
    return V, H


def test_Krylov_Arnoldi():
    """
    Translation of test_Krylov_Arnoldi from MATLAB
    
    Tests the Arnoldi iteration against a fragment of the MATLAB 
    implementation of expokit.
    """
    # System matrix (exact values from MATLAB test)
    A = np.array([
        [0.652541378199936, -0.292928516824949, 0.549087928976002, 0.274632260556064, -0.414286758065495, -0.141241992085791, -0.028985009015485, 0.605937342611256, -0.195349385223348, -0.853531681405666],
        [0.111507396391712, 0.886771403763683, -0.926186402280864, -0.878814748052102, -0.240496674392058, 0.630048246527933, -0.134643721501984, -0.098876392250540, 0.534507822288518, -0.721762559941795],
        [-0.184228225751299, 0.728189979526127, 0.844495081734710, 0.095075956658037, 0.414564407256731, -0.247618105941440, 0.264265765751310, -0.006909214129143, -0.159428715921432, 0.044005920605800],
        [0.384842720632807, -0.427961712778828, 0.159082783046048, -0.217631525890836, 0.281950361270010, -0.782175454527015, 0.295164577068571, 0.228121531073434, 0.360768593258595, 0.343308641521092],
        [0.466710516725629, 0.647902449919577, -0.138568065795660, -0.048407199148869, -0.214935429197563, -0.371866410909703, 0.794448727367410, 0.886239821529420, 0.230418230464495, 0.487226934799377],
        [-0.504441536402227, -0.683930638862332, -0.110954574785176, 0.687389269618629, 0.413797704097389, 0.009111413582124, 0.827318132599109, 0.180993736522104, 0.102327863083104, -0.176443263542827],
        [0.015526934326904, -0.116581152633782, 0.658696622614006, -0.174463584664272, 0.205670047157241, 0.622024548694704, -0.394835060811682, -0.452520224525244, 0.388715632620385, 0.060315005491066],
        [-0.107197579271798, -0.080399191437818, -0.007555629564728, -0.262605107304710, 0.522994075276099, -0.559668119751529, -0.817510097401123, -0.480638527682639, 0.128110533548805, -0.041380923113904],
        [0.268292332294290, 0.714031800806371, 0.395607487326902, 0.400361847002023, 0.525457042575656, 0.216020969859562, -0.425914589519408, -0.031701286989668, -0.399781588176020, -0.805725638821021],
        [0.216736942375567, 0.516814156617457, -0.628881792412746, -0.110508717720818, 0.514734156317458, 0.112609184197350, 0.197728711176581, 0.022365142471784, 0.486695871859436, -0.457708772484572]
    ])
    
    # Set Krylov order
    KrylovOrder = 5
    
    # Input vector (MATLAB uses column vector)
    v = np.ones((len(A), 1))
    
    # Call Arnoldi iteration (Python returns 4 values: V, H, Hlast, happyBreakdown)
    V, H, Hlast, happyBreakdown = arnoldi(A, v, KrylovOrder)
    
    # Call Arnoldi iteration from expokit
    V_expokit, H_expokit = aux_arnoldi_expokit(A, v, KrylovOrder)
    
    # Compare V
    # MATLAB: V_expokit(:, 1:KrylovOrder) - note MATLAB is 1-indexed
    V_expokit_truncated = V_expokit[:, :KrylovOrder]
    maxErr_V = np.max(np.abs(V - V_expokit_truncated))
    assert maxErr_V < 1e-14, f"V comparison failed: max error = {maxErr_V}"
    
    # Compare H
    # MATLAB: H_expokit(1:KrylovOrder, 1:KrylovOrder)
    H_expokit_truncated = H_expokit[:KrylovOrder, :KrylovOrder]
    maxErr_H = np.max(np.abs(H - H_expokit_truncated))
    assert maxErr_H < 1e-14, f"H comparison failed: max error = {maxErr_H}"
    
    # Test passed
    assert True


if __name__ == "__main__":
    test_Krylov_Arnoldi()
    print("test_Krylov_Arnoldi: passed")
