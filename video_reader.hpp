#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <queue>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class video_reader {
public:
    video_reader();
    ~video_reader();
    explicit video_reader(const std::string& video_path);
public:
    [[nodiscard]] inline int get_height() const noexcept               { return df_height_;       };
    [[nodiscard]] inline int get_width() const noexcept                { return df_width_;        };
    [[nodiscard]] inline double get_fps() const noexcept               { return df_fps_;          };
    [[nodiscard]] inline std::string get_pixfmt() const noexcept       { return df_pixfmt_;       };
    [[nodiscard]] inline int get_frames_nb() const noexcept            { return df_frames_nb_;    };
    [[nodiscard]] inline std::string get_video_format() const noexcept { return df_video_format_; };
    [[nodiscard]] inline std::string get_video_codec() const noexcept  { return df_video_codec_;  };
    [[nodiscard]] inline int get_length() const noexcept               { return df_length_;       };
    [[nodiscard]] inline int get_bitrate() const noexcept              { return df_birate_;       };
public:
    int open(const std::string& video_path);
public:
    int read(std::queue<cv::Mat>& frames);
    int read(cv::Mat& frame);
public:
    [[nodiscard]] inline bool is_open() const noexcept { return is_open_; }
private:
    void init() noexcept;
private:
    int read_all_frames();
private:
    AVFormatContext* inctx_;
    AVCodec* vcodec_;
    AVStream* vstrm_;
    SwsContext* sws_ctx_;
    AVPixelFormat dst_pix_fmt_;
private:
    int vstream_idx_;
    int dst_width_;
    int dst_height_;
    bool is_open_;
    std::queue<cv::Mat> all_vframes_;
private:
    std::string df_video_format_;
    std::string df_video_codec_;
    int df_height_;
    int df_width_;
    double df_fps_;
    int df_length_;
    int df_birate_;
    std::string df_pixfmt_;
    int df_frames_nb_;
};

