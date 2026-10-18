#pragma once

#include <set>

#include "util.h"
#include "delaunator.h"

/**
 * delaunay_utils is a wrapper for delaunator which handles translating it into a state and filtering/deduping edges
 */
namespace delaunay_utils {
    inline void add_unique(std::set<std::pair<size_t, size_t>> &uniqueEdges, size_t a, size_t b, size_t c) {
        if (b < a) std::swap(a, b);
        if (c < a) std::swap(a, c);
        if (c < b) std::swap(b, c);
        uniqueEdges.insert({a, b});
        uniqueEdges.insert({b, c});
        uniqueEdges.insert({a, c});
    }

    inline void process_delaunay_edge(ConeOrderingState &state, size_t fromIdx, size_t toIdx, const size_t rn) {
        if (fromIdx > toIdx) std::swap(fromIdx, toIdx);
        if (toIdx < rn) {
            if (dist_sq(state.redPoints[fromIdx], state.redPoints[toIdx]) <= MAX_DIST_BETWEEN_CONES_SQUARED) {
                state.redEdgesAll.push_back({static_cast<int>(fromIdx), static_cast<int>(toIdx)});
            }
        } else if (fromIdx >= rn) {
            if (dist_sq(state.bluePoints[fromIdx - rn], state.bluePoints[toIdx - rn]) <= MAX_DIST_BETWEEN_CONES_SQUARED) {
                state.blueEdgesAll.push_back({static_cast<int>(fromIdx - rn), static_cast<int>(toIdx - rn)});
            }
        }
    }

    inline bool get_all_edges_from_delaunay(ConeOrderingState &state) {
        const int rn = static_cast<int>(state.redPoints.size());
        const int bn = static_cast<int>(state.bluePoints.size());
        std::vector<double> points(rn * 2 + bn * 2);
        for (int i = 0; i < rn; i++) {
            points[i * 2] = state.redPoints[i].x;
            points[i * 2 + 1] = state.redPoints[i].y;
        }
        for (int i = 0; i < bn; i++) {
            points[rn * 2 + i * 2] = state.bluePoints[i].x;
            points[rn * 2 + i * 2 + 1] = state.bluePoints[i].y;
        }
        state.redEdgesAll.clear();
        state.blueEdgesAll.clear();
        if (rn + bn < 3) return false;
        try {
            const delaunator::Delaunator delaunay(points);
            std::set<std::pair<size_t, size_t>> uniqueEdges;
            for (int i = 0; i < static_cast<int>(delaunay.triangles.size()); i += 3) {
                add_unique(uniqueEdges, delaunay.triangles[i], delaunay.triangles[i + 1], delaunay.triangles[i + 2]);
            }
            for (const auto &[fromIdx, toIdx] : uniqueEdges) {
                process_delaunay_edge(state, fromIdx, toIdx, rn);
            }
            return true;
        } catch (...) {
            std::cout << "Bad triangulation\n";
            return false;
        }
    }
}
