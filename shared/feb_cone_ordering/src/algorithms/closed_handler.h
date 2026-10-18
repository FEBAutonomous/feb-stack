#pragma once

#include "util.h"

/**
 * closed_handler determines whether or not the car is in a closed track
 */
namespace closed_handler {
    namespace closed_utils {
        inline void dfsEdges(std::vector<bool> &visited, std::vector<Edge> &curPolygon,
            const std::vector<std::set<int>> &adjacency, const int fromIdx) {
            if (visited[fromIdx]) return;
            visited[fromIdx] = true;
            for (auto &a : adjacency[fromIdx]) {
                if (a > fromIdx) curPolygon.push_back({fromIdx, a});
                dfsEdges(visited, curPolygon, adjacency, a);
            }
        }
        inline bool carContainedInPolygon(const ConeOrderingState &state, const std::vector<Point> &points,
                                          const std::vector<Edge> &edges) {
            std::vector visited(points.size(), false);
            std::vector<std::set<int>> adjacency(points.size());
            for (auto &a : adjacency) { a = std::set<int>(); }
            for (const auto &[fromIdx, toIdx]: edges) {
                if (adjacency[fromIdx].find(toIdx) != adjacency[fromIdx].end()) {
                    adjacency[fromIdx].erase(adjacency[fromIdx].find(toIdx));
                    adjacency[toIdx].erase(adjacency[toIdx].find(fromIdx));
                } else {
                    adjacency[fromIdx].insert(toIdx);
                    adjacency[toIdx].insert(fromIdx);
                }
            }
            std::vector<int> badPoints;
            for (int i = 0; i < static_cast<int>(points.size()); i++) {
                if (static_cast<int>(adjacency[i].size()) == 1) badPoints.push_back(i);
                if (adjacency[i].empty()) visited[i] = true;
            }
            for (int i = 0; i < static_cast<int>(badPoints.size()); i++) {
                for (auto &a : adjacency[badPoints[i]]) {
                    adjacency[a].erase(adjacency[a].find(badPoints[i]));
                    if (static_cast<int>(adjacency[a].size()) == 1) badPoints.push_back(a);
                }
                visited[badPoints[i]] = true;
            }
            std::vector<std::vector<Edge>> relevantEdges;
            for (int i = 0; i < static_cast<int>(points.size()); i++) {
                if (!visited[i]) {
                    std::vector<Edge> curPolygon;
                    dfsEdges(visited, curPolygon, adjacency, i);
                    relevantEdges.push_back(curPolygon);
                }
            }
            for (auto &polygon: relevantEdges) {
                int cnt = 0;
                for (auto &[fromIdx, toIdx]: polygon) {
                    if (intersectsLine(state.car, points[fromIdx], points[toIdx]).first) {
                        cnt++;
                    }
                }
                if (cnt % 2 == 1) return true;
            }
            return false;
        }
    }

    inline bool track_is_closed(ConeOrderingState &state) {
        state.is_closed = closed_utils::carContainedInPolygon(state, state.redPoints, state.redEdgesPolygon)
                          || closed_utils::carContainedInPolygon(state, state.bluePoints, state.blueEdgesPolygon);
        return state.is_closed;
    }
}
