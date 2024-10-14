#include "dehaze.h"
#include <ctime>

// Алгоритм удаления дымки с BGR изображения
cv::Mat dehaze(const cv::Mat& img, int patch_size, int max_iter, double eps, int lambda,
               double tmin, double dp, bool log)
{
    assert(img.channels() == 3 && "not bgr picture");

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    gray = toDuble(gray);

    cv::Vec3b a = get_atm_light(img);
    cv::Scalar a_s(a[0], a[1], a[2]);

    cv::Mat i = im2col<uchar>(img, patch_size, patch_size);
    cv::Mat tmap = tmap_optimal(i, a_s, patch_size, max_iter, eps, log);


    //Удаление блочных артефактов с карты пропускания
    tmap = cv::repeat(tmap, patch_size * patch_size, 1);
    tmap = col2im<double>(tmap, patch_size, patch_size, ceil((double)img.cols / patch_size), ceil((double)img.rows / patch_size));
    tmap = tmap(cv::Range(0, img.rows), cv::Range(0, img.cols));
    tmap = guided_filter(gray, tmap, 30, 1e-4);

    cv::Mat adapt_light = adaptive_atm_light(tmap, img, gray, a_s, lambda);

    cv::Mat res2;

    {
        cv::Mat dimg = toDuble(img);
        cv::Mat temp;
        cv::pow(cv::max(tmap, tmin), dp, temp);
        cv::Mat res = (dimg - adapt_light);
        std::vector<cv::Mat> temp_3ch = { temp, temp, temp };
        cv::merge(temp_3ch, temp);
        res = res / temp + adapt_light;

        cv::threshold(res, res, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(res, res, 1.0, 1.0, cv::THRESH_TRUNC);
        res.convertTo(res, CV_8UC3, 255.);
        res2 = res.clone();
    }


    if (false)
    {
        cv::Mat img2 = cv::imread("/mnt/disk512/work/tools/dehaze/Nuzhny007/rel/orlan/03.45321.jpg");
        cv::Mat dimg = toDuble(img2);
        cv::Mat temp;
        cv::pow(cv::max(tmap, tmin), dp, temp);
        cv::Mat res = (dimg - adapt_light);
        std::vector<cv::Mat> temp_3ch = { temp, temp, temp };
        cv::merge(temp_3ch, temp);
        res = res / temp + adapt_light;

        cv::threshold(res, res, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(res, res, 1.0, 1.0, cv::THRESH_TRUNC);
        res.convertTo(res, CV_8UC3, 255.);
        cv::imwrite("res2.jpg", res);

        cv::Mat adapt_light2;
        adapt_light.convertTo(adapt_light2, CV_8U, 1.);
        cv::imwrite("adapt_light_1.png", adapt_light2);

        adapt_light.convertTo(adapt_light, CV_8U, 255.);
        cv::imwrite("adapt_light_255.png", adapt_light);
    }

    return res2;
}
