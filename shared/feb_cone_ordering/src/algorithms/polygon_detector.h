#pragma once

#include "util.h"

/**
 * polygon_detector finds the relevant polygon from each connected component of edges to use in slope field calculations
 */
namespace polygon_detector {
    inline std::vector<bool> edgeUsed, pointUsed;
    inline std::vector<std::vector<bool>> pointUsed2;
    inline std::vector<std::vector<int>> adjacencyList;

    inline void dfsMarkPointsUsed(const int i) {
        if (pointUsed[i]) return;
        pointUsed[i] = true;
        for (const int j : adjacencyList[i]) {
            dfsMarkPointsUsed(j);
        }
    }

    inline void add_polygon_edges(ConeOrderingState &state, const std::vector<Point> &allPoints, std::vector<Edge> &toPushTo, const int curPoint, int cameFrom) {
        double cameFromAngle;
        if (cameFrom == -1) {
            cameFromAngle = atan2(state.car.p.y - allPoints[curPoint].y, state.car.p.x - allPoints[curPoint].x);
        } else {
            cameFromAngle = atan2(allPoints[cameFrom].y - allPoints[curPoint].y, allPoints[cameFrom].x - allPoints[curPoint].x);
        }
        int bestOther = cameFrom;
        double bestDiff = 2 * M_PI;
        for (auto &a : adjacencyList[curPoint]) {
            if (a != cameFrom) {
                double ang = atan2(allPoints[a].y - allPoints[curPoint].y, allPoints[a].x - allPoints[curPoint].x);
                double diff = angleDiffCCW(cameFromAngle, ang);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestOther = a;
                }
            }
        }
        if (!pointUsed2[bestOther][curPoint]) {
            pointUsed2[bestOther][curPoint] = true;
            toPushTo.push_back({curPoint, bestOther});
            add_polygon_edges(state, allPoints, toPushTo, bestOther, curPoint);
        }
    }

    inline void get_polygon_edges_specific(ConeOrderingState &state, const std::vector<Point> &allPoints, const std::vector<Edge> &allEdges, std::vector<Edge> &toPushTo) {
        toPushTo.clear();
        edgeUsed.clear();
        pointUsed.clear();
        pointUsed2.clear();
        edgeUsed.resize(allEdges.size(), false);
        pointUsed.resize(allPoints.size(), false);
        for (int i = 0; i < static_cast<int>(allPoints.size()); i++) {
            pointUsed2.emplace_back(allPoints.size(), false);
        }
        adjacencyList.clear();
        for (int i = 0; i < static_cast<int>(allPoints.size()); i++) adjacencyList.emplace_back();
        for (auto &[fromIdx, toIdx] : allEdges) {
            adjacencyList[fromIdx].push_back(toIdx);
            adjacencyList[toIdx].push_back(fromIdx);
        }
        int unusedCount = static_cast<int>(allEdges.size());
        while (unusedCount > 0) {
            int closestEdge = -1;
            int endpoint = -1;
            double closestDist = std::numeric_limits<double>::max();
            for (int i = 0; i < static_cast<int>(allEdges.size()); i++) {
                if (!edgeUsed[i]) {
                    auto ln = perpendicular_vector_to_line(allPoints[allEdges[i].fromIdx], allPoints[allEdges[i].toIdx], state.car.p);
                    auto dist = pow(ln.first, 2) + pow(ln.second, 2);
                    if (dist < closestDist) {
                        closestEdge = i;
                        closestDist = dist;
                        double px = ln.first + state.car.p.x;
                        double py = ln.second + state.car.p.y;
                        auto fromPoint = allPoints[allEdges[i].fromIdx];
                        auto toPoint = allPoints[allEdges[i].toIdx];
                        if (pow(px - fromPoint.x, 2) + pow(py - fromPoint.y, 2) < 0.01) {
                            endpoint = 2;
                            continue;
                        }
                        if (pow(px - toPoint.x, 2) + pow(py - toPoint.y, 2) < 0.01) {
                            endpoint = 3;
                            continue;
                        }
                        double ang = atan2(ln.second, ln.first);
                        double ang2 = atan2(fromPoint.y - state.car.p.y, fromPoint.x - state.car.p.x);
                        if (angleDiffCCW(ang2, ang) < M_PI) {
                            endpoint = 0;
                        } else {
                            endpoint = 1;
                        }
                    }
                }
            }
            add_polygon_edges(state, allPoints, toPushTo,
                endpoint % 2 == 0 ? allEdges[closestEdge].fromIdx : allEdges[closestEdge].toIdx,
                endpoint >= 2 ? -1 : endpoint == 0 ? allEdges[closestEdge].toIdx : allEdges[closestEdge].fromIdx);
            dfsMarkPointsUsed(allEdges[closestEdge].fromIdx);
            for (int i = 0; i < static_cast<int>(allEdges.size()); i++) {
                if (!edgeUsed[i] && pointUsed[allEdges[i].fromIdx]) {
                    edgeUsed[i] = true;
                    unusedCount--;
                }
            }
        }
    }

    inline void get_polygon_edges(ConeOrderingState &state) {
        get_polygon_edges_specific(state, state.redPoints, state.redEdgesAll, state.redEdgesPolygon);
        get_polygon_edges_specific(state, state.bluePoints, state.blueEdgesAll, state.blueEdgesPolygon);
    }
}
