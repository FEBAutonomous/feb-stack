"""Bounded selected-cycle joins. Cumulative motion handles skipped cycles."""
import time
from collections import OrderedDict


def identity(meta):
    stamp=meta.header.stamp
    return (meta.run_id,meta.generation,meta.config_id,meta.cycle,stamp.sec,stamp.nanosec)


class CycleJoin:
    def __init__(self,kinds,limit=16):
        self.kinds=tuple(kinds)
        self.limit=limit
        self.pending=OrderedDict()
        self.last_cycle=-1

    @property
    def size(self):
        return len(self.pending)

    def add(self,kind,msg):
        if kind not in self.kinds:
            raise ValueError('unknown join input '+kind)
        cycle=msg.meta.cycle
        if cycle<=self.last_cycle:
            return []
        received,row=self.pending.setdefault(cycle,(time.monotonic(),{}))
        for other in row.values():
            if identity(other.meta)!=identity(msg.meta):
                raise ValueError('selected-cycle identity mismatch')
        row[kind]=msg
        while len(self.pending)>self.limit:
            self.pending.popitem(last=False)
        output=[]
        for cycle,(received,row) in sorted(list(self.pending.items())):
            if all(k in row for k in self.kinds):
                output.append(tuple(row[k] for k in self.kinds))
                self.last_cycle=cycle
        self.pending=OrderedDict((k,v) for k,v in self.pending.items() if k>self.last_cycle)
        return output

    def stale(self,timeout):
        return any(time.monotonic()-receipt>timeout for receipt,_ in self.pending.values())
