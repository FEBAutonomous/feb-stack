#pragma once

#include "util.h"

/**
 * forces handles the updates for the car's pathing by integrating on the slope field, which is a summation of "forces" from each edge
 */
namespace forces {
    inline std::pair<double, double> contribution(const std::vector<Point> &pointList, const Edge &e, const Point &p, const double angle) {
        const auto perp = perpendicular_vector_to_line(
                pointList[e.fromIdx], pointList[e.toIdx], p);
        const double norm = pow(norm_sq(perp), (FORCE_SCALING_POWER + 1)/2);
        if (norm == 0) return {0, 0};
        const double nx = perp.first * cos(angle) - perp.second * sin(angle);
        const double ny = perp.first * sin(angle) + perp.second * cos(angle);
        return {nx / norm, ny / norm};
    }

    inline double get_force_direction_at_point(const ConeOrderingState &state, const Point &p) {
        double tfx = 0;
        double tfy = 0;
        for (auto &edge : state.redEdgesPolygon) {
            auto [cx, cy] = contribution(state.redPoints, edge, p, M_PI * 5/8);
            tfx += cx;
            tfy += cy;
        }
        for (auto &edge : state.blueEdgesPolygon) {
            auto [cx, cy] = contribution(state.bluePoints, edge, p, -M_PI * 5/8);
            tfx += cx;
            tfy += cy;
        }
        return atan2(tfy, tfx);
    }

    inline void update_car_path_once(ConeOrderingState &state) {
        if (state.carPath.empty()) {
            const double ang = get_force_direction_at_point(state, state.car.p);
            state.carPath.push_back({state.car.p, ang});
        } else {
            auto point = state.carPath.back().p;
            auto lastAng = state.carPath.back().angle;
            point.x += AMT_TO_MOVE_CAR_EACH_ITERATION * cos(lastAng);
            point.y += AMT_TO_MOVE_CAR_EACH_ITERATION * sin(lastAng);
            const double newAng = get_force_direction_at_point(state, point);
            state.carPath.push_back({point, newAng});
        }
    }
}
