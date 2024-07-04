This folder has implementations of five filters:

    Constrained extended Kalman filter (CEKF)
    Constrained unscented Kalman filter (CUKF)
    Constrained cubature Kalman filter (CCKF)
    Constrained particle filter (CPF)
    Constrained ensemble Kalman filter (CEnKF)

And three smoothers:

    Constrained extended Kalman smoother (CEKS)
    Constrained unscented Kalman smoother (CUKS)
    Constrained cubature Kalman smoother (CCKS)

Besides respective counterparts without constraints. In addition, four forward-backward filters are defined based on forward filtering passes and backward smoothing passes:

    Constrained extended Kalman filter and smoother (CEKFS)
    Constrained steady extended Kalman filter and smoother (CSEKFS)
    Constrained unscented Kalman filter and smoother (CUKFS)
    Constrained cubature Kalman filter and smoother (CCKFS)

Implemented state estimators were evaluated for a Van de Vusse reactor and a CSTR with ethanol production via glucose fermentation from Zymomonas mobilis. 

Studies carried out with this source code were submitted to Digital Chemical Engineering in a paper entitled "Assessment of forward and forward-backward Bayesian filters"
