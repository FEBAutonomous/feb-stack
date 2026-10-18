#pragma once
// Glue between the team's cone ordering (src/algorithms, unchanged) and the simulator's
// messages: cones as a PoseArray with the colour in orientation.w, the car as a Pose.
#include <cmath>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include "algorithms/util.h"
#include "algorithms/cone_ordering.h"
#include "ordering_robust.h"

namespace ordering_io {
constexpr int BLUE = 1, YELLOW = 2;

inline double yaw_of(const geometry_msgs::msg::Quaternion &q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

/// Fill the ordering state from the car pose and the coloured map, run the ordering. Right on
/// the start gate (four big cones half a metre apart) the slope field can be degenerate and the
/// closed-track integration ends after a couple of steps; then the ordering is run again from a
/// pose a little behind or ahead of the car along its heading, which is the same lap.
inline void run(ConeOrderingState &state, const geometry_msgs::msg::Pose &car, const geometry_msgs::msg::PoseArray &map, bool expect_closed = false) {
    const double yaw = yaw_of(car.orientation);
    const double offsets[] = {0.0, -2.0, 2.0, -4.0, 4.0, -6.0};
    ConeOrderingState best;
    for (double along : offsets) {
        state = ConeOrderingState{};
        state.car = {{car.position.x + along * std::cos(yaw), car.position.y + along * std::sin(yaw)}, yaw};
        for (const auto &p : map.poses) {
            const int colour = static_cast<int>(std::lround(p.orientation.w));
            if (colour == BLUE) state.bluePoints.push_back({p.position.x, p.position.y});
            else if (colour == YELLOW) state.redPoints.push_back({p.position.x, p.position.y});   // the team calls the right side red
        }
        ordering_robust::update(state, 12, expect_closed);
        if (ordering_robust::healthy(state) && (state.is_closed || !expect_closed)) return;
        if (state.redConeOrder.size() > best.redConeOrder.size()) best = state;
    }
    state = best;      // nothing satisfied the caller: the longest ordering found, for the caller to judge
}

inline geometry_msgs::msg::PoseArray to_array(const std::vector<Point> &pts, const std::string &frame) {
    geometry_msgs::msg::PoseArray arr;
    arr.header.frame_id = frame;
    for (const auto &p : pts) {
        geometry_msgs::msg::Pose q;
        q.position.x = p.x;
        q.position.y = p.y;
        q.orientation.w = 1.0;
        arr.poses.push_back(q);
    }
    return arr;
}
}  // namespace ordering_io
