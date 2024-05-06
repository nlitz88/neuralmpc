#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

using std::placeholders::_1;
using namespace std;

class DriveRecorder: public rclcpp::Node
{
    private:

        // filenames for drive and odom, separately
        std::string drive_file_path, odom_file_path;

        // output file handlers
        ofstream drive_file, odom_file;

        // global time tracking variables
        double prev_time = -1.0;
        double global_time_start;

        // subscribers
        rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_subscription_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

        // callback function for drive topic
        void drive_callback(const ackermann_msgs::msg::AckermannDriveStamped::ConstSharedPtr drive_msg) {
            
            // define previous value fields
            static double prev_steering = 0.0;
            static double prev_speed = 0.0;
            static double prev_acceleration = 0.0;

            // extract the fields we want and write them into csv
            double time_sec = drive_msg->header.stamp.sec;
            double time_nsec = drive_msg->header.stamp.nanosec;
            double steering = drive_msg->drive.steering_angle;
            double steering_velocity = drive_msg->drive.steering_angle_velocity;
            double speed = drive_msg->drive.speed;
            double acceleration = drive_msg->drive.acceleration;
            double jerk = drive_msg->drive.jerk;

            // handle these data
            double time_actual = time_sec + time_nsec / 1e9;

            // if it is the first time
            if (prev_time < 0) {
                // set values to zeros
                steering_velocity = 0.0;
                acceleration = 0.0;
                jerk = 0.0;
                // set global start time
                global_time_start = time_actual;
            } else {
                // manually calculate commanded inputs
                if (steering_velocity == 0.0) {
                    steering_velocity = (steering - prev_steering) / (time_actual - prev_time);
                }

                if (acceleration == 0.0) {
                    acceleration = (speed - prev_speed) / (time_actual - prev_time);
                }

                if (jerk == 0.0) {
                    jerk = (acceleration - prev_acceleration) / (time_actual - prev_time);
                }
                
            }

            // record previous data
            prev_time = time_actual;
            prev_steering = steering;
            prev_speed = speed;
            prev_acceleration = acceleration;

            // write them into csv
            drive_file << time_actual - global_time_start << "," << steering << "," << steering_velocity << "," <<
                          speed << "," << acceleration << "," << jerk << "\n";
            RCLCPP_INFO(this->get_logger(), "drive topic recorded");
        }

        // callback function for odom topic
        void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg) {

            // extract the fields we want and write them into csv
            double time_sec = odom_msg->header.stamp.sec;
            double time_nsec = odom_msg->header.stamp.nanosec;
            double x_pos = odom_msg->pose.pose.position.x;
            double y_pos = odom_msg->pose.pose.position.y;
            double z_pos = odom_msg->pose.pose.position.z;
            double q_x = odom_msg->pose.pose.orientation.x;
            double q_y = odom_msg->pose.pose.orientation.y;
            double q_z = odom_msg->pose.pose.orientation.z;
            double q_w = odom_msg->pose.pose.orientation.w;
            double x_velo = odom_msg->twist.twist.linear.x;
            double y_velo = odom_msg->twist.twist.linear.y;
            double z_velo = odom_msg->twist.twist.linear.z;
            double x_ang = odom_msg->twist.twist.angular.x;
            double y_ang = odom_msg->twist.twist.angular.y;
            double z_ang = odom_msg->twist.twist.angular.z;

            // handle these data
            double time_actual = time_sec + time_nsec / 1e9;

            if (prev_time < 0) {

            }
            
            double sinr_cosp = 2 * (q_w * q_x + q_y * q_z);
            double cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y);
            double roll = atan2(sinr_cosp, cosr_cosp);

            double sinp = 2 * (q_w * q_y - q_z * q_x);
            double pitch = asin(sinp);

            double siny_cosp = 2 * (q_w * q_z + q_x * q_y);
            double cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z);
            double yaw = atan2(siny_cosp, cosy_cosp);

            // write them into csv
            odom_file << time_actual - global_time_start << "," << x_pos << "," << y_pos << "," << z_pos << "," <<
                         roll << "," << pitch << "," << yaw << "," << 
                         x_velo << "," << y_velo << "," << z_velo << "," <<
                         x_ang << "," << y_ang << "," << z_ang << "\n";
            RCLCPP_INFO(this->get_logger(), "odom topic recorded");
        }

    public:

        DriveRecorder() : Node("drive_odom_csv_node")
        {
            // declare parameters
            this->declare_parameter("drive_file_path", "/sim_ws/drive_pure_pursuit.csv");
            this->declare_parameter("odom_file_path", "/sim_ws/odom_pure_pursuit.csv");

            // read parameters
            drive_file_path = (this->get_parameter("drive_file_path")).as_string();
            odom_file_path = (this->get_parameter("odom_file_path")).as_string();

            // open these files
            drive_file.open(drive_file_path);
            odom_file.open(odom_file_path);
            RCLCPP_INFO(this->get_logger(), "drive and odom recorder opened");

            // write the first line in
            drive_file << "time(s)" << "," << "steering_angle(rad)" << "," << "steering_angle_velocity(rad/s)" << "," <<
                          "speed(m/s)" << "," << "acceleration(m/s^2)" << "," << "jerk(m/s^3)" << "\n";
            odom_file << "time(s)" << "," << "x_position(m)" << "," << "y_position(m)" << "," << "z_position(m)" << "," <<
                         "roll(rad)" << "," << "pitch(rad)" << "," << "yaw(rad)" << "," << 
                         "x_velocity(m/s)" << "," << "y_velocity(m/s)" << "," << "z_velocity(m/s)" << "," <<
                         "x_angular_velocity(rad/s)" << "," << "y_angular_velocity(rad/s)" << "," << "z_angular_velocity(rad/s)" << "\n";

            // subscribe to the clicked point topic
            drive_subscription_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>("drive", 10,
                                  std::bind(&DriveRecorder::drive_callback, this, _1));
            odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>("ego_racecar/odom", 10,
                                  std::bind(&DriveRecorder::odom_callback, this, _1));
        }

        ~DriveRecorder() {
            // close the file
            drive_file.close();
            odom_file.close();
            RCLCPP_INFO(this->get_logger(), "drive and odom recorder closed");
        }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DriveRecorder>());
    rclcpp::shutdown();
    return 0;
}