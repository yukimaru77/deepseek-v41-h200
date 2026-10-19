#!/usr/bin/env bash
# Exact optimized settings selected by the 2026-10-19 H200 A/B (cand-r3).
# Source via H200_KERNEL_PROFILE=round02; model, API and capacity stay in env.sh.
export TF_EXL3_GLINEAR_NS4=auto4
export TF_EXL3_GLINEAR_NS8=2
export TF_EXL3_GLINEAR_STAGES=4
export TF_EXL3_GLINEAR_TN=1
export TF_EXL3_GLINEAR_DEEP=0
export TF_EXL3_GLINEAR_APF=0
export TF_EXL3_X_DEC=4
export TF_EXL3_X_NS=1
export TF_EXL3_X_MINB=4
export TF_EXL3_X_STAGES=3
export TF_EXL3_PREP_PDL=1
export TF_DS_P2P=1
export TF_DS_P2P_KB=4352
export TF_P2P_SLOTS=4
export TF_P2P_BLOCKS=16
export TF_DS_L2_PREFETCH=0
export TF_DS_HOST13=1
