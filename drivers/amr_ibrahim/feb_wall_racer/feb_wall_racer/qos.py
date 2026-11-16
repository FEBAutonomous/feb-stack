"""Shared DDS policies. Source freshness remains an application responsibility."""
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

SENSOR_QOS=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,history=HistoryPolicy.KEEP_LAST)
QOS=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,history=HistoryPolicy.KEEP_LAST)
LATCHED=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,history=HistoryPolicy.KEEP_LAST)
