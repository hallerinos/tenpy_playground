#!/bin/bash
PROG="$HOME/.conda/envs/tenpy_env/bin/tenpy-run -i DMI_model"
SIM=$1

CMD="$PROG $SIM"
$CMD