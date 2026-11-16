"""Offline registration benchmark on recorded simulator scans.

Truth supplies an independently positioned reference submap and scoring only.
This isolates registration; it is deliberately NOT a full SLAM replay or runtime
localizer. Every prediction, including rejected matches, enters the error totals.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feb_wall_racer.core.slam import WallMap
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.perception import scan_points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('baseline_slam',args.baseline)
    baseline=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    db=sqlite3.connect(str(next(args.bag.glob('*.db3'))))
    topics={i:(name,get_message(kind)) for i,name,kind in db.execute('select id,name,type from topics')}
    scans,positions,headings=[],[],[]
    for topic,t,data in db.execute('select topic_id,timestamp,data from messages order by timestamp'):
        name,kind=topics[topic]
        if not name.endswith(('/lidar','/ips','/imu')):
            continue
        msg=deserialize_message(data,kind)
        if name.endswith('/lidar'):
            scans.append((t,scan_points(msg,DEFAULTS)[0]))
        elif name.endswith('/ips'):
            positions.append((t,msg.x,msg.y))
        else:
            q=msg.orientation
            headings.append((t,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))))
    positions=np.array(positions); headings=np.array(headings)
    headings[:,1]=np.unwrap(headings[:,1])
    stamps=np.array([s[0] for s in scans])
    pose=np.column_stack([np.interp(stamps,positions[:,0],positions[:,i]) for i in [1,2]])
    yaw=np.interp(stamps,headings[:,0],headings[:,1])
    rng=np.random.default_rng(7)
    summary=[]
    for noise in [0.,.015]:
        for bias in [(.0,.0,.03),(.08,.06,.025)]:
            errors={k:[] for k in ['prediction','baseline','planar']}
            accepted=0; ranks=[]; timings=[]
            for i in range(10,min(350,len(scans)),3):
                m=WallMap(DEFAULTS); old=baseline.WallMap(DEFAULTS)
                for j in range(i-6,i):
                    cloud=scans[j][1]
                    m.add(cloud,pose[j],yaw[j],DEFAULTS['lidar_forward'],0.)
                    old.add(cloud,pose[j],yaw[j],DEFAULTS['lidar_forward'],0.)
                c,s=math.cos(yaw[i]),math.sin(yaw[i])
                prior=pose[i]+np.array([[c,-s],[s,c]])@np.array(bias[:2])
                heading=yaw[i]+bias[2]
                cloud=scans[i][1].copy()
                if noise:
                    cloud+=rng.normal(0,noise,cloud.shape)
                    cloud[::20]+=rng.normal(0,.25,cloud[::20].shape)
                old_pose,_,_=old.align(cloud,prior,heading,DEFAULTS['lidar_forward'])
                started=time.perf_counter()
                result=m.align_pose(cloud,prior,heading,DEFAULTS['lidar_forward'])
                timings.append(1000*(time.perf_counter()-started))
                accepted+=result.accepted; ranks.append(result.rank)
                for name,at,angle in [('prediction',prior,heading),('baseline',old_pose,heading),('planar',result.pose,result.yaw)]:
                    error=abs(math.atan2(math.sin(angle-yaw[i]),math.cos(angle-yaw[i])))
                    errors[name].append([float(np.linalg.norm(at-pose[i])),math.degrees(error)])
            summary.append(dict(noise_m=noise,bias=bias,samples=len(timings),accepted=accepted,
                rank_counts={str(k):ranks.count(k) for k in set(ranks)},
                timing_ms=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95))),
                errors={name:dict(mean_position_m=float(np.mean(a,axis=0)[0]),
                    mean_heading_deg=float(np.mean(a,axis=0)[1]),
                    p95_position_m=float(np.percentile(a,95,axis=0)[0]),
                    p95_heading_deg=float(np.percentile(a,95,axis=0)[1])) for name,a in errors.items()}))
    output=dict(bag=str(args.bag),reference='previous six real scans positioned with diagnostic truth; no self insertion',
                limitation='registration benchmark, not an end-to-end SLAM drift measurement',results=summary)
    args.output.write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
