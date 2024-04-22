#!/bin/bash
PROG="$HOME/.conda/envs/tenpy_env/bin/tenpy-run -i chiral_magnet"
SIM=$1

CMD="$PROG $SIM"
$CMD