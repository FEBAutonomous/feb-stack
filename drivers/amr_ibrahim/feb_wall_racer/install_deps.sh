#!/bin/bash
# Installs scipy and CasADi into stack/.pydeps, next to your packages, where the racer's
# launch file finds them. Run once from the host while the FEB box is running (./feb up
# in another terminal). Takes a minute.
set -e
BOX="${FEB_BOX_NAME:-feb-box}"
docker exec "$BOX" bash -c 'pip install --quiet --no-deps --target /home/feb/FEB/feb-racing/stack/.pydeps "scipy==1.11.4" casadi && echo "deps installed in stack/.pydeps"'
