#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <algorithm>
#include "math_features.hpp"


std::vector<double> gaussianKernel(double sigma) {
	const int break_of_sigma = 3;
	const double step = 1.0;
	//инициализация сетки и гауссовского распределения
	double filter_size = break_of_sigma * sigma;
	std::vector<double> grid_x{};
	std::vector<double> gauss_dist{};
	double x_current = -filter_size;
	while (x_current <= filter_size) {
		grid_x.push_back(x_current);
		gauss_dist.push_back(gauss(x_current, sigma));
		x_current += step;
	}

	//Нормализация?
	double sum_tmp = std::reduce(gauss_dist.begin(), gauss_dist.end());
	std::for_each(gauss_dist.begin(), gauss_dist.end(), [=](double& x) {x /= sum_tmp;});

	//Вычисление оператора марра (ядро гауссового фильтра?)
	std::vector<double> kernel{};
	for (int i = 0; i < grid_x.size(); i++) {
		kernel.push_back(marra(grid_x[i], sigma) * gauss_dist[i]);
	}

	sum_tmp = std::reduce(kernel.begin(), kernel.end());
	std::for_each(kernel.begin(), kernel.end(), [&](double& x) {x -= sum_tmp / grid_x.size();});

	sum_tmp = 0;
	for (int i = 0; i < grid_x.size(); i++) {
		sum_tmp += 0.5 * pow(grid_x[i], 2) * kernel[i];
	}
	std::for_each(kernel.begin(), kernel.end(), [=](double& x) {x /= sum_tmp;});

	return kernel;
}


double gauss(double x, double sigma) {
	return 1 / (sqrt(2 * M_PI) * sigma) * exp(-(x * x) / (2 * sigma * sigma));
}


double marra(double x, double sigma) {
	return pow(x, 2) / pow(sigma, 4) - 1 / pow(sigma, 2);
}

