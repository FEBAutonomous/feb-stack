// The team's cone ordering as a simulator node.
//
// Two ways to use it:
//   * live: it listens to /feb/map (cones with the colour in orientation.w) and /feb/pose and
//     publishes the ordered rungs on /feb/cone_order/blue and /feb/cone_order/yellow, index i
//     of one across the track from index i of the other, whenever a map arrives;
//   * on demand: the /feb/order_cones service takes a car pose and a map and returns the rungs
//     and whether the track is closed. The cone racer calls it once, when its map closes.
#include <memory>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "feb_cone_ordering/srv/order_cones.hpp"
#include "ordering_io.h"

using std::placeholders::_1;
using std::placeholders::_2;

class ConeOrderingNode : public rclcpp::Node {
public:
    ConeOrderingNode() : Node("cone_ordering") {
        sub_map_ = create_subscription<geometry_msgs::msg::PoseArray>("/feb/map", 10, std::bind(&ConeOrderingNode::on_map, this, _1));
        sub_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>("/feb/pose", 10, std::bind(&ConeOrderingNode::on_pose, this, _1));
        pub_blue_ = create_publisher<geometry_msgs::msg::PoseArray>("/feb/cone_order/blue", 10);
        pub_yellow_ = create_publisher<geometry_msgs::msg::PoseArray>("/feb/cone_order/yellow", 10);
        service_ = create_service<feb_cone_ordering::srv::OrderCones>("/feb/order_cones", std::bind(&ConeOrderingNode::on_request, this, _1, _2));
        RCLCPP_INFO(get_logger(), "cone ordering up: /feb/order_cones service, /feb/cone_order/* from /feb/map");
    }

private:
    void on_pose(const geometry_msgs::msg::PoseStamped &msg) {
        car_ = msg.pose;
        have_pose_ = true;
    }

    void on_map(const geometry_msgs::msg::PoseArray &msg) {
        if (!have_pose_) return;
        ConeOrderingState state;
        ordering_io::run(state, car_, msg);
        pub_blue_->publish(ordering_io::to_array(state.blueConeOrder, msg.header.frame_id));
        pub_yellow_->publish(ordering_io::to_array(state.redConeOrder, msg.header.frame_id));
    }

    void on_request(const std::shared_ptr<feb_cone_ordering::srv::OrderCones::Request> req,
                    std::shared_ptr<feb_cone_ordering::srv::OrderCones::Response> res) {
        ConeOrderingState state;
        ordering_io::run(state, req->car, req->map, true);     // the racer asks once its map is a closed loop
        res->closed = state.is_closed;
        res->blue = ordering_io::to_array(state.blueConeOrder, req->map.header.frame_id);
        res->yellow = ordering_io::to_array(state.redConeOrder, req->map.header.frame_id);
        RCLCPP_INFO(get_logger(), "ordered %zu blue + %zu yellow cones into %zu rungs (%s track)",
                    state.bluePoints.size(), state.redPoints.size(), state.redConeOrder.size(), state.is_closed ? "closed" : "open");
        if (state.is_closed && state.redConeOrder.size() < 8) RCLCPP_WARN(get_logger(), "ordering collapsed even after flipping suspicious cones; the racer falls back to its own walk");
    }

    geometry_msgs::msg::Pose car_;
    bool have_pose_ = false;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_map_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_blue_, pub_yellow_;
    rclcpp::Service<feb_cone_ordering::srv::OrderCones>::SharedPtr service_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConeOrderingNode>());
    rclcpp::shutdown();
    return 0;
}
