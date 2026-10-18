#pragma once
// A robust front end to the team's ordering: on a closed map a single cone of the wrong colour
// can collapse the ordering to a couple of rungs. When that happens the most suspicious cones
// (nearer to the other colour than to their own) are flipped one at a time and the ordering
// run again; the first result with a real number of rungs wins. The algorithm itself is
// untouched (src/algorithms).
#include <algorithm>
#include <cmath>
#include <vector>

#include "algorithms/util.h"
#include "algorithms/cone_ordering.h"

namespace ordering_robust {

/// Length of a polygon's edges: the lap, roughly, from either boundary.
inline double perimeter(const std::vector<Point> &pts, const std::vector<Edge> &edges) {
    double total = 0.0;
    for (const auto &e : edges) total += std::sqrt(dist_sq(pts[e.fromIdx], pts[e.toIdx]));
    return total;
}

/// A closed track is ordered properly when the rung midpoints go round most of the boundary.
inline bool healthy(const ConeOrderingState &s) {
    if (!s.is_closed) return true;
    const size_t n = std::min(s.redConeOrder.size(), s.blueConeOrder.size());
    if (n < 8) return false;
    double covered = 0.0;
    for (size_t i = 1; i < n; i++) {
        Point a{(s.redConeOrder[i - 1].x + s.blueConeOrder[i - 1].x) / 2, (s.redConeOrder[i - 1].y + s.blueConeOrder[i - 1].y) / 2};
        Point b{(s.redConeOrder[i].x + s.blueConeOrder[i].x) / 2, (s.redConeOrder[i].y + s.blueConeOrder[i].y) / 2};
        covered += std::sqrt(dist_sq(a, b));
    }
    const double lap = std::min(perimeter(s.bluePoints, s.blueEdgesPolygon), perimeter(s.redPoints, s.redEdgesPolygon));
    return covered >= 0.6 * lap;
}

/// Suspicion of every cone: nearest other-colour distance over nearest same-colour distance,
/// smallest first (a cone that sits among the other colour ranks first).
inline std::vector<std::pair<bool, int>> suspects(const std::vector<Point> &blue, const std::vector<Point> &red) {
    std::vector<std::pair<double, std::pair<bool, int>>> scored;
    auto nearest = [](const std::vector<Point> &pts, const Point &p, int skip) {
        double best = 1e9;
        for (int j = 0; j < static_cast<int>(pts.size()); j++) if (j != skip) best = std::min(best, dist_sq(pts[j], p));
        return std::sqrt(best);
    };
    for (int i = 0; i < static_cast<int>(blue.size()); i++) scored.push_back({nearest(red, blue[i], -1) / (nearest(blue, blue[i], i) + 1e-6), {true, i}});
    for (int i = 0; i < static_cast<int>(red.size()); i++) scored.push_back({nearest(blue, red[i], -1) / (nearest(red, red[i], i) + 1e-6), {false, i}});
    std::sort(scored.begin(), scored.end(), [](auto &a, auto &b) { return a.first < b.first; });
    std::vector<std::pair<bool, int>> out;
    for (auto &s : scored) out.push_back(s.second);
    return out;
}

/// Run the ordering; if a closed track came out with almost no rungs, retry with suspicious
/// cones flipped (at most `tries` of them, one at a time). Returns how many flips were needed,
/// or -1 if nothing helped (the caller falls back).
inline int update(ConeOrderingState &state, int tries = 12, bool expect_closed = false) {
    const std::vector<Point> blue = state.bluePoints, red = state.redPoints;
    const Car car = state.car;
    cone_ordering::update(state);
    if (healthy(state) && (state.is_closed || !expect_closed)) return 0;
    const auto order = suspects(blue, red);
    for (int k = 0; k < std::min(tries, static_cast<int>(order.size())); k++) {
        ConeOrderingState trial;
        trial.car = car;
        trial.bluePoints = blue;
        trial.redPoints = red;
        if (order[k].first) { trial.redPoints.push_back(blue[order[k].second]); trial.bluePoints.erase(trial.bluePoints.begin() + order[k].second); }
        else { trial.bluePoints.push_back(red[order[k].second]); trial.redPoints.erase(trial.redPoints.begin() + order[k].second); }
        cone_ordering::update(trial);
        if (healthy(trial) && (trial.is_closed || !expect_closed)) { state = trial; return k + 1; }
    }
    return -1;
}
}  // namespace ordering_robust
