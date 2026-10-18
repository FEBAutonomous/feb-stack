#pragma once

#include <climits>

#include "forces.h"
#include "util.h"

/**
 * projector finds the final cone ordering by calling car path updates until repeated (closed tracks) or max iterations are reached (partial tracks)
 * 
 * computed by:
 * - projecting normally to the car path of length equal to the closest point of a boundary point to the car path
 * - rotating the projection so that it no longer intersects with the previous projection
 * - snapping the rotated points to the closest points on the boundary such that it still doesn't intersect
 * - for partial tracks, projecting infinitely to find when the car left the track
 */
namespace projector {
    inline void get_sides(ConeOrderingState &state, bool go_straight_to_find_track = false) {
        if (state.blueEdgesPolygon.empty() || state.redEdgesPolygon.empty()) return;
        state.blueConeOrder.clear();
        state.redConeOrder.clear();
        state.carPath.clear();
        int numOutsideTrack = 0;
        Car lastRelevantCar {0, 0, 0};
        int numSkipped = 0;

        if (state.is_closed) {
            for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
                forces::update_car_path_once(state);
                bool completed = false;
                for (int i = 0; i < iter - 1; i++) {
                    if (dist_sq(state.carPath.back().p, state.carPath[i].p) < std::pow(AMT_TO_MOVE_CAR_EACH_ITERATION/2, 2)
                        && abs(angleDiffCCW(state.carPath.back().angle, state.carPath[i].angle) - M_PI) > 1 * M_PI/180) {
                        std::rotate(state.carPath.begin(), state.carPath.begin() + i + 1, state.carPath.end());
                        for (int j = 0; j <= i; j++) {
                            state.carPath.pop_back();
                        }
                        completed = true;
                        break;
                    }
                }
                if (completed) break;
            }
        }
        bool last_was_not_in_track = false;
        for (int i = 0; (!state.is_closed && i < NUMBER_ITERATIONS) || (state.is_closed && i < static_cast<int>(state.carPath.size())); i++) {
            if (!state.is_closed) {
                forces::update_car_path_once(state);
                if (go_straight_to_find_track && last_was_not_in_track) state.carPath[i].angle = state.carPath[i-1].angle;
            }
            if ((i - numSkipped) % CONE_ORDER_EVERY != 0) continue;
            bool isFirstOne = state.redConeOrder.empty();
            Point lastRedPoint {0, 0}, lastBluePoint {0, 0};
            if (!isFirstOne) {
                lastRedPoint = state.redConeOrder.back();
                lastBluePoint = state.blueConeOrder.back();
            }

            Car c = {state.carPath[i].p.x, state.carPath[i].p.y, state.carPath[i].angle};

            if (!isFirstOne && isPointBetweenParallelLines(
                c.p, lastRedPoint, lastRelevantCar.p,
                std::atan2(lastRedPoint.y - lastBluePoint.y,
                    lastRedPoint.x - lastBluePoint.x))) {
                numSkipped++;
                continue;
            }

            Car forRed {c.p, c.angle - M_PI_2};
            Car forBlue {c.p, c.angle + M_PI_2};

            bool was_outside = false;
            if (!state.is_closed) {
                // check if it exited the track
                bool isGoodRed = false;
                bool isGoodBlue = false;
                for (auto &[fromIdx, toIdx] : state.redEdgesPolygon) {
                    if (intersectsLine(forRed, state.redPoints[fromIdx], state.redPoints[toIdx]).first) {
                        isGoodRed = true;
                    }
                    if (intersectsLine(forBlue, state.redPoints[fromIdx], state.redPoints[toIdx]).first) {
                        isGoodBlue = true;
                    }
                }
                for (auto &[fromIdx, toIdx] : state.blueEdgesPolygon) {
                    if (intersectsLine(forRed, state.bluePoints[fromIdx], state.bluePoints[toIdx]).first) {
                        isGoodRed = true;
                    }
                    if (intersectsLine(forBlue, state.bluePoints[fromIdx], state.bluePoints[toIdx]).first) {
                        isGoodBlue = true;
                    }
                }
                if (!isGoodRed || !isGoodBlue) was_outside = true;
                else numOutsideTrack = 0;
            }
            last_was_not_in_track = was_outside;
            if (go_straight_to_find_track && was_outside) {
                if (i > 0) {
                    state.carPath[i].angle = state.carPath[i-1].angle;
                } else {
                    state.carPath[i].angle = state.car.angle;
                }
            }

            Point pRed {0, 0};
            double distSqRed = INT_MAX;
            for (auto &[fromIdx, toIdx] : state.redEdgesPolygon) {
                auto ln = perpendicular_vector_to_line(state.redPoints[fromIdx], state.redPoints[toIdx], c.p);
                auto d = pow(ln.first, 2) + pow(ln.second, 2);
                if (d < distSqRed) {
                    distSqRed = d;
                    auto dist = sqrt(distSqRed);
                    pRed = {forRed.p.x + dist * cos(forRed.angle), forRed.p.y + dist * sin(forRed.angle)};
                }
            }
            if (distSqRed >= INT_MAX - 1) break;
            Point pBlue {0, 0};
            double distSqBlue = INT_MAX;
            for (auto &[fromIdx, toIdx] : state.blueEdgesPolygon) {
                auto ln = perpendicular_vector_to_line(state.bluePoints[fromIdx], state.bluePoints[toIdx], c.p);
                auto d = pow(ln.first, 2) + pow(ln.second, 2);
                if (d < distSqBlue) {
                    distSqBlue = d;
                    auto dist = sqrt(distSqBlue);
                    pBlue = {forBlue.p.x + dist * cos(forBlue.angle), forBlue.p.y + dist * sin(forBlue.angle)};
                }
            }
            if (distSqBlue >= INT_MAX - 1) break;

            // If the lines don't intersect with the last line, we just use the perpendicular projection
            // But if they do, we rotate this new line just enough so that it no longer intersects
            if (!isFirstOne && intersectsLine(pRed, pBlue, lastRedPoint, lastBluePoint)) {
                double angDiffRed = angleDiffCCW(
                    atan2(c.p.y - lastRedPoint.y, c.p.x - lastRedPoint.x),
                    atan2(c.p.y - pRed.y, c.p.x - pRed.x));
                if (angDiffRed > M_PI) angDiffRed -= 2 * M_PI;
                double angDiffBlue = angleDiffCCW(
                    atan2(c.p.y - lastBluePoint.y, c.p.x - lastBluePoint.x),
                    atan2(c.p.y - pBlue.y, c.p.x - pBlue.x));
                if (angDiffBlue > M_PI) angDiffBlue -= 2 * M_PI;

                double redAng;
                double blueAng;
                if (std::abs(angDiffRed) < std::abs(angDiffBlue)) {
                    redAng = atan2(lastRedPoint.y - c.p.y, lastRedPoint.x - c.p.x);
                    blueAng = M_PI + redAng;
                } else {
                    blueAng = atan2(lastBluePoint.y - c.p.y, lastBluePoint.x - c.p.x);
                    redAng = M_PI + blueAng;
                }
                pBlue = {c.p.x + sqrt(distSqBlue) * cos(blueAng), c.p.y + sqrt(distSqBlue) * sin(blueAng)};
                pRed = {c.p.x + sqrt(distSqRed) * cos(redAng), c.p.y + sqrt(distSqRed) * sin(redAng)};
            }
            // Now we find the closest points to these projected points which actually are on the delaunay borders
            // But we make sure it doesn't intersect with the last line
            Point newPoint = isFirstOne ? Point({pRed.x, pRed.y}) : lastRedPoint;
            distSqRed = isFirstOne ? INT_MAX : dist_sq(newPoint, pRed);
            for (auto &[fromIdx, toIdx] : state.redEdgesPolygon) {
                auto ln = perpendicular_vector_to_line(state.redPoints[fromIdx], state.redPoints[toIdx], pRed);
                auto d = norm_sq(ln);
                if (d < distSqRed && (isFirstOne || !intersectsLine(c.p, {pRed.x + ln.first,
                    pRed.y + ln.second}, lastRedPoint, lastBluePoint))) {
                    distSqRed = d;
                    newPoint = {pRed.x + ln.first, pRed.y + ln.second};
                }
            }
            pRed = newPoint;

            newPoint = isFirstOne ? Point({pBlue.x, pBlue.y}) : lastBluePoint;
            distSqBlue = isFirstOne ? INT_MAX : dist_sq(newPoint, pBlue);
            for (auto &[fromIdx, toIdx] : state.blueEdgesPolygon) {
                auto ln = perpendicular_vector_to_line(state.bluePoints[fromIdx], state.bluePoints[toIdx], pBlue);
                auto d = norm_sq(ln);
                if (d < distSqBlue && (isFirstOne || !intersectsLine(c.p, {pBlue.x + ln.first,
                    pBlue.y + ln.second}, lastRedPoint, lastBluePoint))) {
                    distSqBlue = d;
                    newPoint = {pBlue.x + ln.first, pBlue.y + ln.second};
                }
            }
            pBlue = newPoint;

            if (!isFirstOne && std::max(dist_sq(pRed, lastRedPoint), dist_sq(pBlue, lastBluePoint))
                < std::pow(AMT_TO_MOVE_CAR_EACH_ITERATION/2, 2)) {
                continue;
            }

            lastRelevantCar = c;
            state.redConeOrder.push_back(pRed);
            state.blueConeOrder.push_back(pBlue);
            if (was_outside) numOutsideTrack++;
        }

        if (!state.is_closed) {
            for (int i = 0; i < numOutsideTrack; i++) {
                state.redConeOrder.pop_back();
                state.blueConeOrder.pop_back();
            }
        } else {
            double closestDist = INT_MAX;
            int closestIdx = -1;
            for (int i = 0; i < static_cast<int>(state.redConeOrder.size()); i++) {
                double d = norm_sq(perpendicular_vector_to_line(state.redConeOrder[i], state.blueConeOrder[i], state.car.p));
                if (d < closestDist) {
                    closestDist = d;
                    closestIdx = i;
                }
            }
            std::rotate(state.redConeOrder.begin(), state.redConeOrder.begin() + closestIdx, state.redConeOrder.end());
            std::rotate(state.blueConeOrder.begin(), state.blueConeOrder.begin() + closestIdx, state.blueConeOrder.end());
        }
        
        if (!go_straight_to_find_track && state.redConeOrder.empty()) get_sides(state, true);
    }
}
