#pragma once

#include <vector>
#include <cmath>
#include <iostream>

// Вычисляет ядро гауссова фильтра (второй порядок)
std::vector<double> gaussianKernel(double sigma);

// Плотность нормального распределения (гауссово)
double gauss(double x, double sigma);

// Лаплассиан гауссиана (оператор Марра)
double marra(double x, double sigma);
