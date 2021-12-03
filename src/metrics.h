double calc_covariance(cv::Mat& img1, cv::Mat& img2, double mean_x,
                       double mean_y) {
  cv::Mat img1_temp;
  cv::Mat img2_temp;
  cv::subtract(img1, mean_x, img1_temp);
  cv::subtract(img2, mean_y, img2_temp);

  return cv::sum(img1_temp.mul(img2_temp))[0] / (img1.rows * img1.cols);
}

double SSIM(cv::Mat& img1, cv::Mat& img2) {
  double k1 = 0.01;
  double k2 = 0.03;
  int L = pow(2, 8) - 1;

  double c1 = (k1 * L) * (k1 * L);
  double c2 = (k2 * L) * (k2 * L);

  double mean_x, mean_y;
  double var_x, var_y;

  cv::Scalar mean, stdev;
  cv::meanStdDev(img1, mean, stdev);
  mean_x = mean[0];
  var_x = stdev[0] * stdev[0];

  cv::meanStdDev(img2, mean, stdev);
  mean_y = mean[0];
  var_y = stdev[0] * stdev[0];

  double covariance = calc_covariance(img1, img2, mean_x, mean_y);

  double ssim =
      ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) /
      ((mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2));
  // std::cout << "cov: " << covariance << std::endl;

  return ssim;
}


double MSE(const cv::Mat& img1, const cv::Mat& img2) {
  const auto width = img1.cols;
  const auto height = img1.rows;

  double sum = 0.0;
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      double diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
      sum += diff * diff;
    }
  }

  double mse = sum / (width * height);
  return mse;
}

double RMSE(const cv::Mat& img1, const cv::Mat& img2) {
  double mse = MSE(img1, img2);
  return sqrt(mse);
}

double PSNR(const cv::Mat& img1, const cv::Mat& img2) {
  int max_px = 255;
  double mse = MSE(img1, img2);

  return (10 * log10((max_px * max_px) / mse));
}