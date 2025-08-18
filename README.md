
ID-EKF vs EKF/UKF/CKF
======================================================================

This project builds a common discrete-time model from a continuous-time SDE using the
Ito–Taylor strong order 1.5 (IT-1.5) scheme (additive noise), then compares EKF, UKF, CKF,
and a placeholder ID-EKF.

Run from `experiments/`:
- python run_armse.py
- python run_nees_nis.py
- python run_stability_grid.py

Results go to `experiments/results/`.
