#include <iostream>
#include <opencv2/opencv.hpp>

#include "metrics.h"

cv::Mat CreateGaussianKernel(int window_size, float sigma = 1.13) {
  cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);

  int half_window_size = window_size / 2;

  // see: lecture_03_slides.pdf, Slide 13
  const double k = 2.5;
  // const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
  // const double sigma = r_max / k;

  // std::cout << sigma << std::endl;

  // sum is for normalization
  float sum = 0.0;

#pragma omp parallel for
  for (int x = -window_size / 2; x <= window_size / 2; x++) {
    for (int y = -window_size / 2; y <= window_size / 2; y++) {
      float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
      kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
      sum += val;
    }
  }

  // normalising the Kernel
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) kernel.at<float>(i, j) /= sum;

  return kernel;
}

void OurFiler_Box(const cv::Mat& input, cv::Mat& output,
                  const int window_size = 5) {
  const auto width = input.cols;
  const auto height = input.rows;

#pragma omp parallel for
  for (int r = window_size / 2; r < height - window_size / 2; ++r) {
    for (int c = window_size / 2; c < width - window_size / 2; ++c) {
      // box filter
      int sum = 0;
      for (int i = -window_size / 2; i <= window_size / 2; ++i) {
        for (int j = -window_size / 2; j <= window_size / 2; ++j) {
          sum += input.at<uchar>(r + i, c + j);
        }
      }
      output.at<uchar>(r, c) = sum / (window_size * window_size);
    }
  }
}

void OurFiler_Gaussian(const cv::Mat& input, cv::Mat& output,
                       const int window_size = 5) {
  const auto width = input.cols;
  const auto height = input.rows;

  cv::Mat gaussianKernel = CreateGaussianKernel(window_size);

#pragma omp parallel for
  for (int r = window_size / 2; r < height - window_size / 2; ++r) {
    for (int c = window_size / 2; c < width - window_size / 2; ++c) {
      int sum = 0;
      for (int i = -window_size / 2; i <= window_size / 2; ++i) {
        for (int j = -window_size / 2; j <= window_size / 2; ++j) {
          sum += input.at<uchar>(r + i, c + j) *
                 gaussianKernel.at<float>(i + window_size / 2,
                                          j + window_size / 2);
        }
      }
      output.at<uchar>(r, c) = sum;
    }
  }
}

void OurFilter_Bilateral(const cv::Mat& input, cv::Mat& output,
                         const int window_size = 5,
                         const double sigma_spectral = 20,
                         const double sigma_spatial = 1.13) {
  const auto width = input.cols;
  const auto height = input.rows;

  //  computing Gaussian kernel

  cv::Mat gaussianKernel = CreateGaussianKernel(window_size, sigma_spatial);

  auto p = [](double spectral_difference, double sigma_spectral) {
    //   compute Gussian
    return std::exp(-spectral_difference / sigma_spectral);
  };

#pragma omp parallel for
  for (int r = window_size / 2; r < height - window_size / 2; ++r) {
    for (int c = window_size / 2; c < width - window_size / 2; ++c) {
      double sum = 0;
      double sum_weights = 0.0;
      const auto input_central = input.at<uchar>(r, c);

      for (int i = -window_size / 2; i <= window_size / 2; ++i) {
        for (int j = -window_size / 2; j <= window_size / 2; ++j) {
          const auto input_other = input.at<uchar>(r + i, c + j);
          double spectral_difference = std::pow(input_other - input_central, 2);
          double weight = p(spectral_difference, sigma_spectral) *
                          gaussianKernel.at<float>(i + window_size / 2,
                                                   j + window_size / 2);

          sum += input.at<uchar>(r + i, c + j) * weight;
          sum_weights += weight;
        }
      }
      output.at<uchar>(r, c) = sum / sum_weights;
    }
  }
}

void OurFilter_JointBilateral(const cv::Mat& input, const cv::Mat& guide,
                              cv::Mat& output, const int window_size = 5,
                              const double sigma_spectral = 20) {
  const auto width = input.cols;
  const auto height = input.rows;

  //  computing Gaussian kernel
  cv::Mat gaussianKernel = CreateGaussianKernel(window_size);

  auto p = [](double spectral_difference, double sigma_spectral) {
    //   compute Gussian
    return std::exp(-spectral_difference / sigma_spectral);
  };

#pragma omp parallel for
  for (int r = window_size / 2; r < height - window_size / 2; ++r) {
    for (int c = window_size / 2; c < width - window_size / 2; ++c) {
      double sum = 0;
      double sum_weights = 0.0;
      const auto input_central = input.at<uchar>(r, c);

      for (int i = -window_size / 2; i <= window_size / 2; ++i) {
        for (int j = -window_size / 2; j <= window_size / 2; ++j) {
          const auto input_other = input.at<uchar>(r + i, c + j);
          double spectral_difference = std::pow(input_other - input_central, 2);
          double weight = p(spectral_difference, sigma_spectral) *
                          gaussianKernel.at<float>(i + window_size / 2,
                                                   j + window_size / 2);

          sum += guide.at<uchar>(r + i, c + j) * weight;
          sum_weights += weight;
        }
      }
      output.at<uchar>(r, c) = sum / sum_weights;
    }
  }
}

cv::Mat Upsample(const cv::Mat& rgb, const cv::Mat& depth) {
  int factor = log2(rgb.rows / depth.rows);
  cv::Mat original = rgb.clone();
  cv::Mat upsampled = depth.clone();

  for (int i = 0; i < factor; ++i) {
    cv::resize(upsampled, upsampled, upsampled.size() * 2);
    cv::resize(original, original, upsampled.size());
    OurFilter_JointBilateral(original, upsampled, upsampled, 5, 0.1);
  }
  cv::resize(upsampled, upsampled, rgb.size());

  OurFilter_JointBilateral(original, upsampled, upsampled, 5, 0.1);
  return upsampled;
}

int main(int argc, char** argv) {
  cv::Mat im = cv::imread("../data/lena.png", 0);

  cv::Mat light = cv::imread("../data/light_pie.png", 0);
  cv::Mat dark = cv::imread("../data/dark_pie.png", 0);

  if (im.data == nullptr) {
    std::cerr << "Failed to load image" << std::endl;
  }

  cv::Mat noise(im.size(), im.type());
  uchar mean = 0;
  uchar stddev = 25;
  cv::randn(noise, mean, stddev);

  cv::Mat noisy = im + noise;
  // rgb += noise;

  cv::Mat opencv_out, upsample_out;

  // bilateral
  double window_size = 11;
  cv::bilateralFilter(noisy, opencv_out, window_size, 2 * window_size,
                      window_size / 2);
  cv::imwrite("../out/opencv_bilateral.png", opencv_out);

  cv::Mat bilateral_out =
      cv::Mat::zeros(opencv_out.rows, opencv_out.cols, CV_8UC1);
  cv::Mat joint_out = cv::Mat::zeros(opencv_out.rows, opencv_out.cols, CV_8UC1);

  OurFilter_Bilateral(noisy, bilateral_out, 5, 2000);
  cv::imwrite("../out/our_bilateral.png", bilateral_out);

  // Subtask 1 - START
  // float sigma_spatial[4] = {0.1, 1.13, 10, 20};
  // float sigma_spectral[4] = {0.2, 20, 200, 2000};

  // std::cout << "[spect::spat],mse,rmse,psnr,ssim" << std::endl;

  // for (int i = 0; i < 4; i++) {
  //   for (int j = 0; j < 4; j++) {
  //     OurFilter_Bilateral(noisy, bilateral_out, 5, sigma_spectral[j],
  //                         sigma_spatial[i]);
  //     cv::imwrite("../out/our_bilateral_" + std::to_string(j) + "_" +
  //                     std::to_string(i) + ".png",
  //                 bilateral_out);

  //     double mse = MSE(im, bilateral_out);
  //     double rmse = RMSE(im, bilateral_out);
  //     double psnr = PSNR(im, bilateral_out);
  //     double ssim = SSIM(im, bilateral_out);

  //     std::cout << "[" << std::to_string(sigma_spectral[j]) << "," <<
  //     std::to_string(sigma_spatial[i]) << ","
  //               << rmse << "," << psnr << "," << ssim << "],"
  //               << std::endl;
  //   }
  // }

  // Subtask 1 - END

  // OurFilter_JointBilateral(dark, light, joint_out, 5, 20);
  // cv::imwrite("../out/our_joint.png", joint_out);

  std::string files[12] = {"aloe",       "baby1", "bowling1", "cloth4",
                           "flowerpots", "midd1", "book7",    "book2",
                           "book3",      "book4", "book5",    "book6"};

  for (int i = 0; i < 12; i++) {
    std::string rgb_name = "../data/12/" + files[i] + "_rgb.png";
    std::string gt_depth_name = "../data/12/" + files[i] + "_depth.png";
    std::string depth_name = "../data/12/" + files[i] + "_depth_small.png";
    std::string out_name = "../out/" + files[i] + "_upsampled.png";
    std::string diff_name = "../out/" + files[i] + "_diff.png";
    std::string diff_name_cv_nn = "../out/" + files[i] + "_cvnndiff.png";
    std::string cv_name = "../out/" + files[i] + "_cv.png";
    std::string cv_nn_name = "../out/" + files[i] + "_cv_nn.png";
    std::string diff_name_cv = "../out/" + files[i] + "_cvdiff.png";
    cv::Mat rgb = cv::imread(rgb_name, 0);
    cv::Mat depth = cv::imread(depth_name, 0);
    cv::Mat gt_depth = cv::imread(gt_depth_name, 0);

    // std::cout << gt_depth.size() << "," << depth.size() << std::endl;

    auto t_begin_cv = std::chrono::high_resolution_clock::now();
    cv::Mat opencv_bilinear(gt_depth.size(), gt_depth.type());
    cv::resize(depth, opencv_bilinear, gt_depth.size(), 0, 0, cv::INTER_LINEAR);
    auto t_end_cv = std::chrono::high_resolution_clock::now();

    auto duration_cv =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end_cv -
                                                                  t_begin_cv);

    cv::imwrite(cv_name, opencv_bilinear);

    auto t_begin_cv_nn = std::chrono::high_resolution_clock::now();
    cv::Mat opencv_nearest(gt_depth.size(), gt_depth.type());
    cv::resize(depth, opencv_nearest, gt_depth.size(), 0, 0, cv::INTER_NEAREST);
    auto t_end_cv_nn = std::chrono::high_resolution_clock::now();

    auto duration_cv_nn =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            t_end_cv_nn - t_begin_cv_nn);

    cv::imwrite(cv_nn_name, opencv_nearest);

    auto t_begin = std::chrono::high_resolution_clock::now();
    upsample_out = Upsample(rgb, depth);
    auto t_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
        t_end - t_begin);

    cv::imwrite(out_name, upsample_out);
    // std::cout << gt_depth.size() << "," << upsample_out.size() << std::endl;
    cv::imwrite(diff_name, gt_depth - upsample_out);
    cv::imwrite(diff_name_cv, gt_depth - opencv_bilinear);
    cv::imwrite(diff_name_cv_nn, gt_depth - opencv_nearest);

    double mse = MSE(gt_depth, upsample_out);
    double rmse = RMSE(gt_depth, upsample_out);
    double psnr = PSNR(gt_depth, upsample_out);
    double ssim = SSIM(gt_depth, upsample_out);

    double mse_cv = MSE(gt_depth, opencv_bilinear);
    double rmse_cv = RMSE(gt_depth, opencv_bilinear);
    double psnr_cv = PSNR(gt_depth, opencv_bilinear);
    double ssim_cv = SSIM(gt_depth, opencv_bilinear);

    double mse_cv_nn = MSE(gt_depth, opencv_nearest);
    double rmse_cv_nn = RMSE(gt_depth, opencv_nearest);
    double psnr_cv_nn = PSNR(gt_depth, opencv_nearest);
    double ssim_cv_nn = SSIM(gt_depth, opencv_nearest);

    std::cout << files[i] << "," << rmse << "," << psnr << "," << ssim << ","
              << duration.count() << "," << duration_cv.count() << ","
              << duration_cv_nn.count() << "," << rmse_cv << "," << psnr_cv
              << "," << ssim_cv << "," << rmse_cv_nn << "," << psnr_cv_nn << ","
              << ssim_cv_nn << std::endl;
  }
  return 0;
}