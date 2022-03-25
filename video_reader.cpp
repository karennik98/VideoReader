#include "video_reader.hpp"

#include <opencv2/imgproc.hpp>

video_reader::video_reader()
        : inctx_(nullptr), vcodec_(nullptr), vstrm_(nullptr), sws_ctx_(nullptr), vstream_idx_(0), dst_width_(0),
          dst_height_(0), is_open_(false), dst_pix_fmt_(AV_PIX_FMT_BGR24) {
    init();
}

video_reader::video_reader(const std::string &video_path)
        : video_reader() {
    open(video_path);
}

int video_reader::open(const std::string &video_path) {
    int ret = avformat_open_input(&inctx_, video_path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR]: Fail to avforamt_open_input(\"" << video_path << "\"): ret=" << ret << std::endl;
        return -1;
    }

    ret = avformat_find_stream_info(inctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR]: Fail to avformat_find_stream_info: ret=" << ret << std::endl;
        return -1;
    }

    ret = av_find_best_stream(inctx_, AVMEDIA_TYPE_VIDEO, -1, -1, &vcodec_, 0);
    if (ret < 0) {
        std::cerr << "[ERROR]: Fail to av_find_best_stream: ret=" << ret << std::endl;
        return -1;
    }

    vstream_idx_ = ret;
    vstrm_ = inctx_->streams[vstream_idx_];

    ret = avcodec_open2(vstrm_->codec, vcodec_, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR]: Fail to avcodec_open2: ret=" << ret << std::endl;
        return -1;
    }

    df_video_format_ = inctx_->iformat->name;
    df_video_codec_ = vcodec_->name;
    df_height_ = vstrm_->codec->height;
    df_width_ = vstrm_->codec->width;
    df_fps_ = av_q2d(vstrm_->codec->framerate);
    df_length_ = av_rescale_q(vstrm_->duration, vstrm_->time_base, {1, 1000}) / 1000.;
    df_pixfmt_ = av_get_pix_fmt_name(vstrm_->codec->pix_fmt);
    df_frames_nb_ = vstrm_->nb_frames;

#ifdef __DEBUG__
    // print input video stream informataion
    std::cout
            << "video_path: " << video_path << "\n"
            << "format: " << inctx_->iformat->name << "\n"
            << "vcodec: " << vcodec_->name << "\n"
            << "size:   " << vstrm_->codec->width << 'x' << vstrm_->codec->height << "\n"
            << "fps:    " << av_q2d(vstrm_->codec->framerate) << " [fps]\n"
            << "length: " << av_rescale_q(vstrm_->duration, vstrm_->time_base, {1,1000}) / 1000. << " [sec]\n"
            << "pixfmt: " << av_get_pix_fmt_name(vstrm_->codec->pix_fmt) << "\n"
            << "frame:  " << vstrm_->nb_frames << "\n"
            << std::flush;
#endif

    dst_width_ = vstrm_->codec->width;
    dst_height_ = vstrm_->codec->height;
    sws_ctx_ = sws_getCachedContext(
            nullptr, vstrm_->codec->width, vstrm_->codec->height,
            vstrm_->codec->pix_fmt,
            dst_width_, dst_height_, dst_pix_fmt_,
            SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        std::cerr << "[ERROR]: Fail to sws_getCachedContext" << std::endl;
        return -1;
    }

#ifdef __DEBUG__
    std::cout << "[INFO]: output: " << dst_width_ << 'x' << dst_height_ << ',' << av_get_pix_fmt_name(dst_pix_fmt_) << std::endl;
#endif
    read_all_frames();
    is_open_ = true;
    return ret;
}

void video_reader::init() noexcept {
}

int video_reader::read(cv::Mat &frame) {
    if (!all_vframes_.empty()) {
        frame = all_vframes_.front();
        all_vframes_.pop();
        return 1;
    }
    return -1;
}

int video_reader::read(std::queue <cv::Mat> &frames) {
    if (!all_vframes_.empty()) {
        frames = all_vframes_;
        while (!all_vframes_.empty()) {
            all_vframes_.pop();
        }
        return 1;
    }
    return -1;
}

int video_reader::read_all_frames() {
    AVFrame *frame = av_frame_alloc();
    std::vector <uint8_t> framebuf(avpicture_get_size(dst_pix_fmt_, dst_width_, dst_height_));
    avpicture_fill(reinterpret_cast<AVPicture *>(frame), framebuf.data(), dst_pix_fmt_,
                   dst_width_, dst_height_);

    // decoding loop
    AVFrame *decframe = av_frame_alloc();
    unsigned nb_frames = 0;
    bool end_of_stream = false;
    int got_pic = 0;
    AVPacket pkt;
    do {
        if (!end_of_stream) {
            // read packet from input file
            int ret = av_read_frame(inctx_, &pkt);
            if (ret < 0 && ret != AVERROR_EOF) {
                std::cerr << "[ERROR]: Fail to av_read_frame: ret=" << ret << std::endl;
                return -1;
            }
            if (ret == 0 && pkt.stream_index != vstream_idx_) {
                av_free_packet(&pkt);
                continue;
            }
            end_of_stream = (ret == AVERROR_EOF);
        }
        if (end_of_stream) {
            // null packet for bumping process
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;
        }
        // decode video frame
        avcodec_decode_video2(vstrm_->codec, decframe, &got_pic, &pkt);
        if (!got_pic) {
            av_free_packet(&pkt);
            continue;
        }
        // convert frame to OpenCV matrix
        sws_scale(sws_ctx_, decframe->data, decframe->linesize, 0, decframe->height,
                  frame->data, frame->linesize);
        {
            cv::Mat image(dst_height_, dst_width_, CV_8UC3, framebuf.data(), frame->linesize[0]);
            all_vframes_.push(image.clone());
        }
#ifdef __DEBUG__
        std::cout << "[INFO]: nb_frames: " << nb_frames << '\r' << std::flush;  // dump progress
#endif
        ++nb_frames;
        av_free_packet(&pkt);
    } while (!end_of_stream || got_pic);

    av_frame_free(&decframe);
    av_frame_free(&frame);

    return 0;
}

video_reader::~video_reader() {
    avcodec_close(vstrm_->codec);
    avformat_close_input(&inctx_);
}

