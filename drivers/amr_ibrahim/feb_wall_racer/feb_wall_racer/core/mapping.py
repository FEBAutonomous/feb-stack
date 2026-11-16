"""Mapping/localization state; registration kernels remain unchanged."""
import math
import logging
import numpy as np
from .slam import WallMap
from .perception import corridor_is_usable


class MappingOperations:
    def _integrate(self, speed, dt, count_mapping_distance):
        if self.odom_position is None:
            return
        if self.last_odom_position is None:
            self.last_odom_position = self.odom_position.copy()
            self.last_integrated_heading_correction = self.heading_correction
            return
        displacement = self.odom_position - self.last_odom_position
        self.last_odom_position = self.odom_position.copy()
        previous = self.last_integrated_heading_correction
        difference = self.heading_correction-previous
        angle = previous+.5*math.atan2(math.sin(difference),math.cos(difference))
        c,s = math.cos(angle),math.sin(angle)
        displacement = np.array([c*displacement[0]-s*displacement[1],
                                 s*displacement[0]+c*displacement[1]])
        self.last_integrated_heading_correction = self.heading_correction
        self.pose += displacement
        ds = max(float(speed), 0.0) * dt
        if count_mapping_distance:
            self.travelled += ds

    def _mapping_step(self, stamp, observation):
        points = np.vstack((observation.left, observation.right))
        if len(self.slam.start_scan) == 0 and len(points) >= 12:
            self.slam.start_scan = points.copy()
        if self.travelled - self.last_keyframe_distance >= self.p["keyframe_distance"]:
            if len(self.slam.points) >= self.slam.min_matches:
                alignment = self.slam.align_pose(
                    points, self.pose, self.yaw, self.p["lidar_forward"])
                self.last_matches, self.last_residual = alignment.matches, alignment.residual
                if alignment.accepted:
                    difference = math.atan2(math.sin(alignment.yaw-self.yaw),
                                            math.cos(alignment.yaw-self.yaw))
                    self.lidar_heading_offset += difference
                    self.heading_correction += difference
                    self.last_integrated_heading_correction = self.heading_correction
                    self.pose = alignment.pose.copy()
                    self.yaw = alignment.yaw
                    self.last_loc_t = stamp
            if self.travelled - self.last_keyframe_distance >= self.p["keyframe_distance"]:
                seed_labels = (
                    len(observation.centre) >= self.p["corridor_map_seed_samples"]
                    and observation.confidence >= self.p["corridor_map_seed_confidence"]
                )
                self.slam.add(
                    points, self.pose, self.yaw, self.p["lidar_forward"], self.travelled,
                    left_points=observation.left if seed_labels else None,
                    right_points=observation.right if seed_labels else None,
                    label_weight=1.0 if seed_labels else 0.0,
                )
                self.last_keyframe_distance = self.travelled
        self._append_path(observation.width)

        closed, residual = self.slam.loop_match(
            points, self.pose, self.start_pose, self.yaw, self.start_yaw, self.travelled, self.p)
        if closed:
            correction = self.pose - self.start_pose
            if np.linalg.norm(correction) <= self.p["max_loop_correction"]:
                self._finish_mapping(correction)
            else:
                self._controlled_stop("loop closure correction %.2f m exceeds limit"
                                      % np.linalg.norm(correction))
        elif self.p["debug"] and stamp - self.last_status_t > 2.0:
            self.get_logger().info("mapping: %.1f m, %d path samples, %d map points, loop residual %.3f"
                                   % (self.travelled, len(self.path_points), len(self.slam.points), residual))

    def _append_path(self, width):
        if not self.path_points or np.linalg.norm(self.pose - self.path_points[-1]) >= self.p["path_sample_distance"]:
            self.path_points.append(self.pose.copy())
            self.path_widths.append(float(width))
            self.last_path_distance = self.travelled


class SlamEngine(MappingOperations):
    def __init__(self,p,logger=None):
        self.p=p
        self.logger=logger or logging.getLogger(__name__)
        self.reset()

    def get_logger(self):
        return self.logger

    def reset(self):
        self.slam=WallMap(self.p)
        self.pose=np.zeros(2)
        self.yaw=self.heading_correction=self.lidar_heading_offset=0.
        self.last_integrated_heading_correction=0.
        self.odom_position=self.last_odom_position=None
        self.start_pose=self.start_yaw=None
        self.travelled=self.last_keyframe_distance=self.last_path_distance=0.
        self.path_points=[]
        self.path_widths=[]
        self.last_loc_t=0.
        self.last_scan_t=None
        self.last_status_t=-1e9
        self.last_matches=0
        self.last_residual=float('inf')
        self.last_match_odom_stamp=None
        self.last_match_accepted=False
        self.mode='MAPPING'
        self.stop_reason=''
        self.map_id=self.map_revision=self.pose_revision=0

    def _controlled_stop(self,reason):
        self.stop_reason=reason
        self.mode='STOPPED'

    def process(self,motion,observation,stamp):
        self.last_scan_t=stamp
        self.odom_position=np.asarray(motion.odom_position,dtype=float)
        self.yaw=math.atan2(math.sin(motion.yaw+self.lidar_heading_offset),
                            math.cos(motion.yaw+self.lidar_heading_offset))
        self.heading_correction=motion.heading_correction+self.lidar_heading_offset
        if self.start_pose is None:
            self.start_pose=self.pose.copy()
            self.start_yaw=self.yaw
            self.last_odom_position=self.odom_position.copy()
            self.last_integrated_heading_correction=self.heading_correction
        else:
            self._integrate(motion.body_speed,motion.dt,False)
        if not self.slam.frozen:
            self.travelled=float(motion.travelled)
        self.pose_revision+=1
        if observation.front_clearance<self.p['obstacle_stop_range']:
            return False
        if self.mode=='MAPPING':
            if corridor_is_usable(observation,self.p):
                previous=self.last_keyframe_distance
                self._mapping_step(stamp,observation)
                if self.last_keyframe_distance!=previous:
                    self.map_revision+=1
            elif self.path_points:
                # Gap guidance can keep moving without a reliable width.
                # Keep the driven trail continuous for closed-track planning.
                self._append_path(float('nan'))
        if self.slam.frozen and self.mode!='STOPPED':
            odom_stamp=(motion.odom_stamp.sec,motion.odom_stamp.nanosec)
            if self.last_match_odom_stamp is None or odom_stamp>self.last_match_odom_stamp:
                # Several scans can select the same odometry snapshot.
                # Apply at most one correction to that prediction.
                self.last_match_odom_stamp=odom_stamp
                wall_points=np.vstack((observation.left,observation.right))
                corrected,matches,residual=self.slam.align(wall_points,self.pose,self.yaw,self.p['lidar_forward'])
                self.last_matches,self.last_residual=matches,residual
                self.last_match_accepted=matches>=self.slam.min_matches and residual<.40
                if self.last_match_accepted:
                    shift=corrected-self.pose
                    norm=float(np.linalg.norm(shift))
                    if norm>.15:
                        shift*=.15/norm
                    self.pose+=shift
                    self.last_loc_t=stamp
            age=stamp-self.last_loc_t
            if age>self.p['stop_after']:
                self._controlled_stop('wall-map localization lost')
            return self.last_match_accepted and age<=self.p['lost_after']
        return False

    def _finish_mapping(self,correction):
        if not self.path_points:
            return
        self.path_points=[point-min(i/max(len(self.path_points)-1,1),1.)*correction
                          for i,point in enumerate(self.path_points)]
        self.slam.close_loop(correction,max(self.travelled,1e-6))
        self.pose=self.start_pose.copy()
        self.map_id+=1
        self.map_revision+=1
        self.mode='FINALIZING'
        self.last_loc_t=self.last_scan_t or 0.
