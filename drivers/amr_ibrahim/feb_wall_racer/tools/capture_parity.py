"""Record public stack inputs/outputs for behavior comparison, in a ROS shell."""
import argparse
import os
from pathlib import Path
import signal
import subprocess

TOPICS = ['/scan','/odom','/imu','/clock','/feb/state','/feb/mission/live',
          '/drive','/ackermann_cmd','/feb/pose','/feb/status','/feb/raceline',
          '/feb/trajectory','/feb/mpc_prediction','/feb/wall/map/left',
          '/feb/wall/map/right','/feb/run/context','/feb/perception/walls',
          '/feb/estimation/motion','/feb/slam/state','/feb/slam/track_map',
          '/feb/slam/wall_context','/feb/planning/race_plan',
          '/feb/planning/local_reference','/feb/control/prediction']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--duration',type=float,default=120.)
    args=parser.parse_args()
    if args.duration<=0 or args.output.exists():
        parser.error('duration must be positive and output must not exist')
    proc=subprocess.Popen(['ros2','bag','record','-o',str(args.output),*TOPICS],start_new_session=True)
    try:
        proc.wait(timeout=args.duration)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGINT)
        proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGINT)
            proc.wait(timeout=20)


if __name__=='__main__':
    main()
