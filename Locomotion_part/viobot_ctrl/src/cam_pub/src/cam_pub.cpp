#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <poll.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

class MJPEGCameraNode : public rclcpp::Node
{
public:
  MJPEGCameraNode() : Node("mjpeg_camera_node"), fd_(-1), is_running_(false)
  {
    topic_name_ = this->declare_parameter<std::string>(
      "topic_name", "/camera_x001/image_raw/compressed");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "camera_link");
    video_device_ = this->declare_parameter<std::string>("video_device", "/dev/video0");
    image_width_ = this->declare_parameter<int>("image_width", 1920);
    image_height_ = this->declare_parameter<int>("image_height", 1080);
    buffer_count_ = this->declare_parameter<int>("buffer_count", 8);
    qos_depth_ = this->declare_parameter<int>("qos_depth", 10);
    dqbuf_timeout_ms_ = this->declare_parameter<int>("dqbuf_timeout_ms", 2000);
    max_consecutive_timeouts_ = this->declare_parameter<int>("max_consecutive_timeouts", 5);

    if (!validate_parameters()) {
      exit_code_ = 1;
      return;
    }

    rclcpp::QoS qos_profile(static_cast<size_t>(qos_depth_));
    qos_profile.reliable();
    publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      topic_name_, qos_profile);

    if (!init_camera()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to init camera. Node will exit for service restart.");
      exit_code_ = 1;
      return;
    }

    is_running_ = true;
    capture_thread_ = std::thread(&MJPEGCameraNode::capture_loop, this);
  }

  ~MJPEGCameraNode()
  {
    is_running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    stop_camera();
  }

  bool is_initialized() const
  {
    return is_initialized_;
  }

  int exit_code() const
  {
    return exit_code_;
  }

private:
  struct Buffer {
    void* start = nullptr;
    size_t length = 0;
  };

  int fd_ = -1;
  int image_width_ = 1920;
  int image_height_ = 1080;
  int buffer_count_ = 8;
  int qos_depth_ = 10;
  int dqbuf_timeout_ms_ = 2000;
  int max_consecutive_timeouts_ = 5;
  std::string topic_name_;
  std::string frame_id_;
  std::string video_device_;
  std::vector<Buffer> buffers_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;

  std::thread capture_thread_;
  std::atomic<bool> is_running_;
  std::atomic<bool> is_initialized_{false};
  std::atomic<int> exit_code_{0};

  bool validate_parameters()
  {
    if (topic_name_.empty() || frame_id_.empty() || video_device_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "topic_name, frame_id and video_device must be non-empty.");
      return false;
    }
    if (image_width_ <= 0 || image_height_ <= 0) {
      RCLCPP_ERROR(this->get_logger(), "Invalid image size: %dx%d", image_width_, image_height_);
      return false;
    }
    if (buffer_count_ < 2) {
      RCLCPP_WARN(this->get_logger(), "buffer_count=%d is too small; using 2.", buffer_count_);
      buffer_count_ = 2;
    }
    if (qos_depth_ < 1) {
      RCLCPP_WARN(this->get_logger(), "qos_depth=%d is too small; using 1.", qos_depth_);
      qos_depth_ = 1;
    }
    if (dqbuf_timeout_ms_ < 500) {
      RCLCPP_WARN(this->get_logger(), "dqbuf_timeout_ms=%d is too small; using 500.", dqbuf_timeout_ms_);
      dqbuf_timeout_ms_ = 500;
    }
    if (max_consecutive_timeouts_ < 1) {
      RCLCPP_WARN(
        this->get_logger(),
        "max_consecutive_timeouts=%d is too small; using 1.",
        max_consecutive_timeouts_);
      max_consecutive_timeouts_ = 1;
    }
    return true;
  }

  bool init_camera()
  {
    fd_ = open(video_device_.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to open device %s: %s",
        video_device_.c_str(),
        strerror(errno));
      return false;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = image_width_;
    fmt.fmt.pix.height = image_height_;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set format: %s", strerror(errno));
      stop_camera();
      return false;
    }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
      RCLCPP_ERROR(this->get_logger(), "Camera does not support MJPEG format!");
      stop_camera();
      return false;
    }
    if (fmt.fmt.pix.width != static_cast<uint32_t>(image_width_) ||
      fmt.fmt.pix.height != static_cast<uint32_t>(image_height_))
    {
      RCLCPP_WARN(
        this->get_logger(),
        "Camera adjusted image size from %dx%d to %ux%u.",
        image_width_,
        image_height_,
        fmt.fmt.pix.width,
        fmt.fmt.pix.height);
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = static_cast<uint32_t>(buffer_count_);
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to request buffers: %s", strerror(errno));
      stop_camera();
      return false;
    }
    if (req.count < 2) {
      RCLCPP_ERROR(this->get_logger(), "Camera returned too few buffers: %u", req.count);
      stop_camera();
      return false;
    }

    buffers_.resize(req.count);

    for (size_t i = 0; i < req.count; i++) {
      struct v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;

      if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
        RCLCPP_ERROR(this->get_logger(), "VIDIOC_QUERYBUF failed: %s", strerror(errno));
        stop_camera();
        return false;
      }

      buffers_[i].length = buf.length;
      buffers_[i].start = mmap(
        nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

      if (buffers_[i].start == MAP_FAILED) {
        RCLCPP_ERROR(this->get_logger(), "mmap failed: %s", strerror(errno));
        buffers_[i].start = nullptr; // 重置防止析构时误用
        stop_camera();
        return false;
      }

      if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        RCLCPP_ERROR(this->get_logger(), "VIDIOC_QBUF failed: %s", strerror(errno));
        stop_camera();
        return false;
      }
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
      RCLCPP_ERROR(this->get_logger(), "VIDIOC_STREAMON failed: %s", strerror(errno));
      stop_camera();
      return false;
    }

    is_initialized_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "Camera started successfully: device=%s topic=%s size=%ux%u timeout=%dms max_timeouts=%d",
      video_device_.c_str(),
      topic_name_.c_str(),
      fmt.fmt.pix.width,
      fmt.fmt.pix.height,
      dqbuf_timeout_ms_,
      max_consecutive_timeouts_);
    return true;
  }

  void capture_loop()
  {
    int consecutive_timeouts = 0;

    while (rclcpp::ok() && is_running_) {
      struct pollfd camera_poll;
      memset(&camera_poll, 0, sizeof(camera_poll));
      camera_poll.fd = fd_;
      camera_poll.events = POLLIN;

      const int poll_result = poll(&camera_poll, 1, dqbuf_timeout_ms_);
      if (poll_result == 0) {
        consecutive_timeouts++;
        RCLCPP_WARN(
          this->get_logger(),
          "No camera frame for %d ms (%d/%d).",
          dqbuf_timeout_ms_,
          consecutive_timeouts,
          max_consecutive_timeouts_);
        if (consecutive_timeouts >= max_consecutive_timeouts_) {
          fail_and_shutdown("Camera frame timeout threshold reached.");
          break;
        }
        continue;
      }
      if (poll_result < 0) {
        if (errno == EINTR) {
          continue;
        }
        fail_and_shutdown(std::string("poll failed: ") + strerror(errno));
        break;
      }
      if (camera_poll.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        fail_and_shutdown("Camera poll reported device error/hangup.");
        break;
      }
      if (!(camera_poll.revents & POLLIN)) {
        continue;
      }

      struct v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN || errno == EINTR) {
          continue;
        }
        fail_and_shutdown(std::string("DQBUF failed: ") + strerror(errno));
        break;
      }

      consecutive_timeouts = 0;

      sensor_msgs::msg::CompressedImage msg;
      msg.header.stamp = this->now();
      msg.header.frame_id = frame_id_;
      msg.format = "jpeg";

      msg.data.resize(buf.bytesused);
      memcpy(msg.data.data(), buffers_[buf.index].start, buf.bytesused);

      publisher_->publish(msg);

      if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        fail_and_shutdown(std::string("VIDIOC_QBUF failed: ") + strerror(errno));
        break;
      }
    }
  }

  void fail_and_shutdown(const std::string& reason)
  {
    RCLCPP_ERROR(this->get_logger(), "%s Exiting so systemd can restart cam_pub.", reason.c_str());
    exit_code_ = 2;
    is_running_ = false;
    rclcpp::shutdown();
  }

  void stop_camera()
  {
    if (fd_ < 0) return;

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);

    for (auto& b : buffers_) {
      if (b.start && b.start != MAP_FAILED) {
        munmap(b.start, b.length);
        b.start = nullptr;
      }
    }
    buffers_.clear();

    close(fd_);
    fd_ = -1;
    is_initialized_ = false;
    RCLCPP_INFO(this->get_logger(), "Camera stopped and cleaned up.");
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MJPEGCameraNode>();
  if (!node->is_initialized()) {
    rclcpp::shutdown();
    return node->exit_code();
  }

  rclcpp::spin(node);
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return node->exit_code();
}
