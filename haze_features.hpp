#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


 
/* Функция, рассчитывает контрастную энергию для трёх цветовых каналов изображения: 
серого, сине - жёлтого, красно - зелёного
* @param image - входное BGR изображение
*/
std::vector<cv::Mat> contrastEnergy(const cv::Mat& image);

/* Функция, рассчитывает энтропию изображения, принимает серый канал изображения
* @param gray входный серый канал изображения
* @param patch_size размер локального участка
*/
double imageEntropy(const cv::Mat& gray, int patch_size);

/* Функция, считает стандартное отклонение
* @param gray входный серый канал изображения
* @param gaussKernel ядро фильтра Гаусса
* @param mu мю
*/
double stdDeviation(const cv::Mat& gray, const cv::Mat& gaussKernel, double mu);

/* Функция, считает нормализованную дисперсию
*/
double normDisp(double std, double mu);

/* Поиск оптимальной карты пропускания
* @param patches входное изображение разделённое по блокам в виде столбцов
* @param a координата атмосферного света
* @param patch_size размер блока
* @param max_iter максимальное количество итераций в алгоритме Нелдера Мида
* @param eps точность в алгоритме Нелдера Мида
* @param log вывод процесса
*/
cv::Mat tmap_optimal(cv::Mat& patches, cv::Scalar a, int patch_size, int max_iter, double eps, bool log);