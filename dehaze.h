#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include "atm_light.h"
#include "utils.h"
#include "haze_features.hpp"
#include "filtering.hpp"

/* Алгоритм по удалению дымки
* @param img входное BGR изображение
* @param pathc_size размер окна, на которые изображение разбивается при обработке
* @param max_iter максимальное количество итераций для алгоритма Нелдера Мида
* @param eps точность вычислений алгоритма Нелдера Мида
* @param lambda коэффициент при вычислении адаптивного атмосферного света
* @param tmin ограничение снизу на величину пропускания
* @param dp dehaze power коэффициент при убирании дымки
* @param log вывод процесса составления оптимальной карты пропускания
*/
cv::Mat dehaze(const cv::Mat& img, int path_size = 16, int max_iter = 10e5, double eps = 10e-7,
			   int lamda = 4, double tmin = 0.2, double dp = 0.7, bool log = false);