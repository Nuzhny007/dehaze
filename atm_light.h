#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/* Вычисляет точку атмосферного света методом QuadTree division. BGR изображение
* @param img входное изображение
* @param stopdiv_size относительные размер конечного участка
* @param minfilt_size радиус действий минимального фильтра
*/
cv::Vec3b get_atm_light(const cv::Mat& img, double stopdiv_size = 0.2, int minfilt_size = 5);

/* Адаптивный атмосферный свет
* @param tmap карта пропускания
* @param img входное BGR изображение
* @param gray входное серое изображение
* @param atm_light коордианата атмосферного света
* @param lambda коэффициент
*/
cv::Mat adaptive_atm_light(const cv::Mat& tmap, const cv::Mat& img, const cv::Mat& gray, cv::Scalar atm_light, int lambda);