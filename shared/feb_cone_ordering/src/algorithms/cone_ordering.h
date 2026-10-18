#pragma once

#include "delaunay_utils.h"
#include "polygon_detector.h"
#include "projector.h"
#include "closed_handler.h"

/**
 * cone_ordering handles the overall update for cone ordering on a state, executing all subtasks consecutively
 */
namespace cone_ordering {
    inline void update(ConeOrderingState &state) {
        if (delaunay_utils::get_all_edges_from_delaunay(state)) {
            polygon_detector::get_polygon_edges(state);
            closed_handler::track_is_closed(state);
            projector::get_sides(state);
        }
    }
}
