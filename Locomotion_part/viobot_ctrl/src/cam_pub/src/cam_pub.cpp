#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>

class MJPEGCameraNode : public rclcpp::Node
{
public:
  MJPEGCameraNode() : Node("mjpeg_camera_node"), fd_(-1), is_running_(false)
  {
    rclcpp::QoS qos_profile(10);
    qos_profile.reliable();
    publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "/camera_x001/image_raw/compressed", qos_profile);

    if (!init_camera("/dev/video0", 1920, 1080, 8)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to init camera. Node will not publish.");
      return;
    }

    // 启动独立的采集线程
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

private:
  struct Buffer {
    void* start = nullptr;
    size_t length = 0;
  };

  int fd_ = -1;  // 必须初始化为 -1
  std::vector<Buffer> buffers_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
  
  std::thread capture_thread_;
  std::atomic<bool> is_running_;

  bool init_camera(const char* device, int width, int height, int buffer_count)
  {
    fd_ = open(device, O_RDWR);
    if (fd_ < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open device: %s", strerror(errno));
      return false;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set format: %s", strerror(errno));
      stop_camera(); // 失败时清理
      return false;
    }

    // 校验硬件是否接受了 MJPEG 格式
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
      RCLCPP_ERROR(this->get_logger(), "Camera does not support MJPEG format!");
      stop_camera();
      return false;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = buffer_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to request buffers: %s", strerror(errno));
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

    RCLCPP_INFO(this->get_logger(), "Camera started successfully");
    return true;
  }

  // 独立的死循环线程，与 ROS 2 Executor 解耦
  void capture_loop()
  {
    while (rclcpp::ok() && is_running_) {
      struct v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      // 阻塞等待相机出帧
      if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN || errno == EINTR) {
          continue; // 非致命错误，重试
        }
        RCLCPP_ERROR(this->get_logger(), "DQBUF failed: %s. Camera disconnected?", strerror(errno));
        break; // 遇到致命错误（如拔出设备）退出循环
      }

      sensor_msgs::msg::CompressedImage msg;
      msg.header.stamp = this->now();
      msg.header.frame_id = "camera_link";
      msg.format = "jpeg";

      msg.data.resize(buf.bytesused);
      memcpy(msg.data.data(), buffers_[buf.index].start, buf.bytesused);

      publisher_->publish(msg);

      if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        RCLCPP_ERROR(this->get_logger(), "Failed to requeue buffer!");
        break;
      }
    }
  }

  void stop_camera()
  {
    if (fd_ < 0) return;

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type); // 关闭流

    for (auto& b : buffers_) {
      if (b.start && b.start != MAP_FAILED) {
        munmap(b.start, b.length);
        b.start = nullptr;
      }
    }
    buffers_.clear();

    close(fd_);
    fd_ = -1;
    RCLCPP_INFO(this->get_logger(), "Camera stopped and cleaned up.");
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MJPEGCameraNode>());
  rclcpp::shutdown();
  return 0;
}