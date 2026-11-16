"""Compare deterministic exported behavior fixtures; do not hide mismatches."""
import argparse
import json
from pathlib import Path
import math


def differences(left,right,path='',atol=1e-8):
    if isinstance(left,dict) and isinstance(right,dict):
        if left.keys()!=right.keys():
            yield path+': fields differ'
        for key in left.keys()&right.keys():
            yield from differences(left[key],right[key],path+'/'+key,atol)
    elif isinstance(left,list) and isinstance(right,list):
        if len(left)!=len(right):
            yield path+': lengths differ'
        for i,(a,b) in enumerate(zip(left,right)):
            yield from differences(a,b,path+'/'+str(i),atol)
    elif isinstance(left,(float,int)) and isinstance(right,(float,int)) and not isinstance(left,bool) and not isinstance(right,bool):
        if not (left==right or math.isfinite(left) and math.isfinite(right) and abs(left-right)<=atol):
            yield path+': numeric values differ'
    elif left!=right:
        yield path+': values differ'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path); p.add_argument('candidate',type=Path)
    args=p.parse_args()
    failed=list(differences(json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text())))
    print(json.dumps({'passed':not failed,'differences':failed},indent=2))
    raise SystemExit(bool(failed))


if __name__=='__main__':
    main()
