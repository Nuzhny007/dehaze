#include <functional>

#include "haze_features.hpp"
#include "utils.h"
#include "math_features.hpp"
#include "atm_light.h"
#include "minimizer.h"

/* Приводит ядро к нужному виду для свёртки
*/
cv::Point prepareKernel(const cv::Mat& kernel_in, cv::Mat& kernel_out) {
    cv::flip(kernel_in, kernel_out, -1);
    cv::Point anchor(kernel_out.cols - kernel_out.cols / 2 - 1, kernel_out.rows - kernel_out.rows / 2 - 1);

    return anchor;
}


/* Функция, рассчитывает контрастную энергию для конкретного канала
@param image_clr входной канал
@param out_ce_clr выходная контрастная энергия
@param sigma средне квадратичное отклонение в фильтре гаусса
@param t пороговое значения для подавления шума
@param k коэффициент насыщения (semisaturatoin ?)
*/
void channelContrastEnergy(const cv::Mat& image_clr, cv::Mat& out_ce_clr, double sigma, double t, double k) {
    const int border_size = 20;
    cv::Mat bordered_img = make_border(image_clr, border_size);

    std::vector<double> kernel = gaussianKernel(sigma);
    cv::Mat y_kernel(kernel);
    cv::Mat x_kernel, conv_x, conv_y;
    cv::transpose(y_kernel, x_kernel);

    cv::Mat flipped_x_kernel, flipped_y_kernel;
    cv::Point anchorx = prepareKernel(x_kernel, flipped_x_kernel);
    cv::Point anchory = prepareKernel(y_kernel, flipped_y_kernel);
    cv::filter2D(bordered_img, conv_x, -1, flipped_x_kernel, anchorx, 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(bordered_img, conv_y, -1, flipped_y_kernel, anchory, 0.0, cv::BORDER_CONSTANT);

    cv::Mat z_bordered;
    cv::sqrt(conv_x.mul(conv_x) + conv_y.mul(conv_y), z_bordered);
    cv::Mat z = remove_border(z_bordered, border_size);

    double z_max{};
    cv::minMaxIdx(z, nullptr, &z_max);

    cv::Mat ce_clr_temp = (z_max * z) / (z + z_max * k) - t;
    cv::threshold(ce_clr_temp, out_ce_clr, 0.0000001, 0, cv::THRESH_TOZERO);

}

// Функция, рассчитывает контрастную энергию для трёх цветовых каналов изображения
std::vector<cv::Mat> contrastEnergy(const cv::Mat& image_in) {
    assert(image_in.channels() == 3 && "not bgr image");

    const double sigma = 3.25; //средне квадратичное отклонение в фильтре гаусса
    const double k = 0.1;
    const double t_gray = 9.225496406318721e-4 * 255;
    const double t_by = 8.969246659629488e-4 * 255;
    const double t_rg = 2.069284034165411e-4 * 255;
    const int border_size = 20;

    cv::Mat image;
    image_in.convertTo(image, CV_64F);
    std::vector<cv::Mat> rgb_cnls;
    cv::split(image, rgb_cnls);

    cv::Mat gray_cnl = 0.299 * rgb_cnls[2] + 0.587 * rgb_cnls[1] + 0.114 * rgb_cnls[0];
    cv::Mat by_cnl = 0.5 * rgb_cnls[2] + 0.5 * rgb_cnls[1] - rgb_cnls[0];
    cv::Mat rg_cnl = rgb_cnls[2] - rgb_cnls[1];

    std::vector<cv::Mat> image_cnls = { gray_cnl, by_cnl, rg_cnl };
    std::vector<cv::Mat> out_ce(3);
    std::vector<double> ce_t_coef = { t_gray, t_by, t_rg };
    for (int i = 0; i < out_ce.size(); i++) {
        channelContrastEnergy(image_cnls[i], out_ce[i], sigma, ce_t_coef[i], k);
    }

    return out_ce;
}

// рассчитывает энтропию изображения, принимает серый канал изображения
double imageEntropy(const cv::Mat& gray, int patch_size) {
    assert(gray.channels() == 1 && "not gray image");

    cv::Mat hist;
    calc_cnl_hist(gray, hist);
    hist = hist / pow(patch_size, 2);

    double entropy = 0;
    for (int i = 0; i < hist.rows; i++) {
        double log_part = hist.at<float>(i, 0) == 0 ? 0 : log2(hist.at<float>(i, 0));
        entropy += hist.at<float>(i, 0) * log_part;
    }

    return -entropy;
}

// считает стандартное отклонение
double stdDeviation(const cv::Mat& gray, const cv::Mat& gaussKernel, double mu) {
    assert(gray.channels() == 1 && "not gray image");

    cv::Mat temp = (gray - mu);
    cv::pow(temp, 2, temp);
    temp = temp.mul(gaussKernel);
    cv::sqrt(temp, temp);

    return cv::sum(temp)[0];
}

// считает нормализованную дисперсию
double normDisp(double std, double mu) {
    return std / (mu);
}

// Количественная оценка дымки в локальном участке изображения
double tmap(cv::Vec<double, 1> x, cv::Mat& patch, cv::Scalar a, int patch_size, cv::Mat& gaussianKernel) {
    double t = x[0];

    cv::Mat j;
    patch.convertTo(j, CV_64F);
    j = (j - a) / t + a;
    j.convertTo(j, CV_8U);
    cv::Mat jgray;
    cv::cvtColor(j, jgray, cv::COLOR_BGR2GRAY);
    std::vector<int> shape = { patch_size, patch_size };
    cv::Mat square_j = j.reshape(3, shape);
    cv::transpose(square_j, square_j);
    std::vector<cv::Mat> contrast_energy = contrastEnergy(square_j);
    double entropy = imageEntropy(jgray, patch_size);
    jgray.convertTo(jgray, CV_64F);
    jgray = jgray / 255;
    double mu = cv::sum(jgray.mul(gaussianKernel))[0];
    double stddev = stdDeviation(jgray, gaussianKernel, mu);
    double norm = normDisp(stddev, mu);

    std::vector<double> params = {
        entropy,
        cv::mean(contrast_energy[0])[0],
        cv::mean(contrast_energy[1])[0],
        cv::mean(contrast_energy[2])[0],
        stddev,
        norm
    };

    double res = 1;
    for (auto p : params) {
        res *= log(1 + abs(p));
    }

    res = res == 0.0 ? 0.00000001 : res;

    return -res;
}

// Поиск оптимальной карты пропускания
using namespace std::placeholders;
cv::Mat tmap_optimal(cv::Mat& patches, cv::Scalar a, int patch_size, int max_iter, double eps, bool log)
{
    cv::Mat t_opt = cv::Mat::zeros(1, patches.cols, CV_64F);
    cv::Mat gauss = cv::getGaussianKernel(patch_size, patch_size / 4, CV_64F);
    cv::mulTransposed(gauss, gauss, false);
    std::vector<int> shape = { patch_size * patch_size, 1 };
    cv::Mat sheped_gauss = gauss.reshape(1, shape);

    // Params for Nelder_Mead_Optimizer
    double step = 0.1;
    double no_improve_thr = 10e-6;
    int no_improv_break = 10;
    double alpha = 1;
    double gamma = 2;
    double rho = -0.5;
    double sigma = 0.5;

    if (log)
        std::cout << "patches = " << patches.cols << std::endl;

    shape = { patches.rows * patches.channels() };
#pragma omp parallel for
    for (int patch = 0; patch < patches.cols; ++patch)
    {
        //std::cout << "patch: " << patch << std::endl;
        cv::Mat i = patches(cv::Range(0, patches.rows), cv::Range(patch, patch + 1));
        double min_val{};
        cv::Mat tmp_patch;
        i.convertTo(tmp_patch, CV_64F);
        tmp_patch = tmp_patch / a;
        cv::minMaxIdx(tmp_patch.reshape(1, shape), &min_val);

        cv::Vec<double, 1> xmin(1-min_val);
        auto tmap_binded = bind(tmap, _1, i, a, patch_size, sheped_gauss);

        auto res = Nelder_Mead_Optimizer<decltype(tmap_binded), 1>(tmap_binded, xmin, step, no_improve_thr, no_improv_break, max_iter, alpha, gamma, rho, sigma, log);

        if (log)
            std::cout << patch << ": " << xmin[0] << std::endl;

        t_opt.at<double>(0, patch) = res[0];

    }

    return t_opt;
}
