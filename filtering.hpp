#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/* Минимальный фильтр. Уменьшает интенсивность белых объектов. 
Используется как предобработка для вычисления атмосферной освещённости
* @param img входное изображение
* @param out_img выходное изображение
* @param size радиус действия
*/
void min_filter(const cv::Mat& img, cv::Mat& out_img, int size);

/* GuidedFilter уточняет карту пропускания, опираясь на исоходное изображение
* @param img входное изображение
* @param p карта пропускания
* @param r коэффициент
* @param eps точность
*/
cv::Mat guided_filter(const cv::Mat& img, const cv::Mat& p, int r, double eps);