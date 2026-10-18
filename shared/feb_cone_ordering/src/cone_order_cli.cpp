// Run the cone ordering on a text file, for tests and for looking at what it does to a map.
//   input (stdin), one item per line:   car X Y YAW      b X Y      y X Y
//   output (stdout):                    closed 0|1       then one "rung BX BY YX YY" per rung
#include <iostream>
#include <sstream>
#include <string>

#include "algorithms/util.h"
#include "algorithms/cone_ordering.h"
#include "ordering_robust.h"

int main() {
    ConeOrderingState state;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        if (kind == "car") {
            in >> state.car.p.x >> state.car.p.y >> state.car.angle;
        } else if (kind == "b" || kind == "y") {
            Point p{};
            in >> p.x >> p.y;
            (kind == "b" ? state.bluePoints : state.redPoints).push_back(p);
        }
    }
    const int flips = ordering_robust::update(state);
    std::cout << "flips " << flips << "\n";
    std::cout << "closed " << (state.is_closed ? 1 : 0) << "\n";
    for (size_t i = 0; i < state.redConeOrder.size() && i < state.blueConeOrder.size(); i++) {
        std::cout << "rung " << state.blueConeOrder[i].x << " " << state.blueConeOrder[i].y << " "
                  << state.redConeOrder[i].x << " " << state.redConeOrder[i].y << "\n";
    }
    return 0;
}
