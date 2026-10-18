#pragma once

#include <vector>
#include <chrono>
#include <cmath>
#include <map>

/**
 * This file is dedicated to constants and helper functions
 * Includes functions for calculating geometric intersections & projections
 */

//region CONSTANTS
// Maximum distance between adjacent cones (to tune)
constexpr int MAX_DIST_BETWEEN_CONES = 10;
// Amount to move the car each iteration while integrating slope field (to tune)
constexpr double AMT_TO_MOVE_CAR_EACH_ITERATION = 0.1;
// Maximum number of iterations when integrating for partial tracks (keep at 2000)
constexpr int NUMBER_ITERATIONS = 2000;
// Maximum number of iterations when integrating for closed tracks (should be approaching infinity)
constexpr int MAX_ITERATIONS = 10000;
// Ratio of number of projections / number of iterations of integration (2 is good, increase if want to lessen computation)
constexpr int CONE_ORDER_EVERY = 2;
// Power to scale distance by when computing contribution to slope field (keep at 2)
constexpr double FORCE_SCALING_POWER = 2;
// If true, viz shows cone ordering in order via rainbow; if false, shows blue/red lines for cone ordering
constexpr bool DRAW_RAINBOW_ORDER = true;

// For readability
constexpr int MAX_DIST_BETWEEN_CONES_SQUARED = MAX_DIST_BETWEEN_CONES * MAX_DIST_BETWEEN_CONES;
//endregion

//region STRUCTURES
struct Point {
    double x, y; // math coordinates (+y up)
};

inline Point operator-(const Point& a, const Point& b) {
    return {a.x - b.x, a.y - b.y};
}

struct Edge {
    int fromIdx;
    int toIdx;
};

struct Car {
    Point p;   // math coordinates
    double angle;  // in radians, w.r.t positive x
};

struct ConeOrderingState {
    // Input into cone ordering
    Car car;
    std::vector<Point> redPoints, bluePoints;

    // Computed by cone ordering
    std::vector<Edge> redEdgesAll, blueEdgesAll;
    std::vector<Edge> redEdgesPolygon, blueEdgesPolygon;
    std::vector<Car> carPath;
    std::vector<Point> redConeOrder, blueConeOrder;

    bool is_closed;
};
//endregion

//region UTIL FUNCTIONS
inline double dist_sq(const Point &a, const Point &b) {
    return pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
}

inline double norm_sq(const Point &a) {
    return pow(a.x, 2) + pow(a.y, 2);
}

inline double norm_sq(const std::pair<double, double> &a) {
    return pow(a.first, 2) + pow(a.second, 2);
}

inline double norm_sq(const double a, const double b) {
    return pow(a, 2) + pow(b, 2);
}

inline double angleDiffCCW(const double ang1, const double ang2) {
    double ret = ang1 - ang2;
    while (ret < 0) ret += 2 * M_PI;
    while (ret >= 2 * M_PI) ret -= 2 * M_PI;
    return ret;
}

inline std::pair<double, double> perpendicular_vector_to_line(const Point &a, const Point &b, const Point &p) {
    const double ABx = b.x - a.x;
    const double ABy = b.y - a.y;
    const double APx = p.x - a.x;
    const double APy = p.y - a.y;
    double t = (ABx * APx + ABy * APy) / (ABx * ABx + ABy * ABy);
    t = std::max(0., std::min(t, 1.));
    const double Qx = a.x + t * ABx;
    const double Qy = a.y + t * ABy;
    return {Qx - p.x, Qy - p.y};
}

inline std::pair<bool, double> intersectsLine(const Car& A, const Point& B, const Point& C) {
    // Direction of the infinite line from A
    const double dx = cos(A.angle);
    const double dy = sin(A.angle);

    // Segment BC direction
    const double sx = C.x - B.x;
    const double sy = C.y - B.y;

    // Solve for intersection:
    // A + t*(dx,dy) = B + u*(sx,sy)
    // => t*(dx,dy) - u*(sx,sy) = (B - A)

    const double rx = B.x - A.p.x;
    const double ry = B.y - A.p.y;

    const double det = dx * (-sy) - dy * (-sx); // determinant

    // Parallel or coincident
    if (std::fabs(det) < 1e-12)
        return {false, 0.0};

    // Solve for t and u
    double t = (rx * (-sy) - ry * (-sx)) / det;
    const double u = (dx * ry - dy * rx) / det;

    // Segment intersects line if 0 <= u <= 1
    return {u > 1e-12 && u < 1-1e-12 && t > 0, t};
}

inline bool intersectsLine(const Point &L11, const Point &L12, const Point &L21, const Point &L22) {
    auto intersection = intersectsLine({L11, std::atan2(L12.y - L11.y, L12.x - L11.x)}, L21, L22);
    if (!intersection.first) return false;
    return intersection.second * intersection.second < dist_sq(L11, L12)-1e-12;
}

inline double dot(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y;
}

inline bool isPointBetweenParallelLines(
    const Point& p,
    const Point& line1Point,
    const Point& line2Point,
    double angle,
    double eps = 1e-9
) {
    // Normal vector to the lines
    Point n = {-std::sin(angle), std::cos(angle)};

    double d1 = dot(p - line1Point, n);
    double d2 = dot(p - line2Point, n);

    // Check if 0 lies between d1 and d2
    return (d1 >= -eps && d2 <= eps) || (d1 <= eps && d2 >= -eps);
}
//endregion

/**
 * profiler allows easy profiling of various parts of the program
 */
namespace profiler {
    inline std::map<std::string, long> durations;

    template <typename F>
    long long profile(F func, const std::string& id, const bool accumulate) {
        const auto start = std::chrono::steady_clock::now();
        func();
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (accumulate) durations[id] += duration;
        else durations[id] = duration;
        return duration;
    }

    inline long long get_duration(const std::string& id) {
        return durations[id];
    }

    inline void print_duration(const std::string& id) {
        std::cout << "Duration of " << id << ": " << get_duration(id) << " ms\n";
    }
}
