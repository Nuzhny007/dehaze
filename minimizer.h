#ifndef NELDER_MEAD_NELDERMEADOPTIMIZER_H
#define NELDER_MEAD_NELDERMEADOPTIMIZER_H

#include <vector>
#include <tuple>
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

template<int dim> using Tuple = std::tuple<cv::Vec<double, dim>, double>;

template<int dim> using Vector = cv::Vec<double, dim>;
/**
* Nelder Mean function.
*
* This function compute the nelder_meand on your func as loss function.
*
* @param f  function to minimize, must return a double scalar.
* @param x_start  initial position.
* @param step  look-around radius in initial step.
* @param no_improv_thr  threshold on improve classification .
* @param no_improv_break  break after no_improv_break iterations without improvement.
* @param max_iter  break after exeed max_iter iterations.
* @param alpha  function to minimize, must return a double scalar.
* @param gamma  function to minimize, must return a double scalar.
* @param rho  function to minimize, must return a double scalar.
* @param sigma  function to minimize, must return a double scalar.

* @return best x find.
*/


template <typename T, int dim> cv::Vec<double, dim> Nelder_Mead_Optimizer(T func, cv::Vec<double, dim> x_start, double step = 0.1,
                                                                          double no_improve_thr = 10e-6, int no_improv_break = 10, int max_iter = 200, double alpha = 1, double gamma = 2,
                                                                          double rho = -0.5, double sigma = 0.5, bool log = false)
{
    double best, prev_best = func(x_start);
    int no_improv = 0;
    std::vector<Tuple<dim>> result;

    result.push_back(std::make_tuple(x_start, prev_best));

    for (int i = 0; i < dim; i++) {
        Vector<dim> x(x_start);
        x[i] += step;
        result.push_back({ x, func(x) });
    }

    for (int iteration = 0;; ++iteration)
    {
        // order
        std::sort(result.begin(), result.end(), [](const Tuple<dim>& a, const Tuple<dim>& b) -> bool
        {
            return (std::get<1>(a) < std::get<1>(b));
        });

        best = std::get<1>(result[0]);

        // break after max_iter
        if (max_iter && iteration >= max_iter)
        {
            std::cout << "max_iter: " << iteration << std::endl;
            return std::get<0>(result[0]);
        }

        //break after no_improv_break iterations with no improvement
        if (log)
            std::cout << "... best so far:  " << best << std::endl;

        if (best < (prev_best - no_improve_thr))
        {
            no_improv = 0;
            prev_best = best;
        }
        else
        {
            no_improv++;
        }

        if (no_improv >= no_improv_break)
        {
            if (log)
                std::cout << "no improve: " << iteration << " (" << no_improv_break << ")\n";

            return std::get<0>(result[0]);
        }

        //centroid
        Vector<dim> centroid_pt = Vector<dim>::all(0);
        for (auto it_pt = result.begin(); it_pt != (result.end() - 1); it_pt++)
        {
            centroid_pt += std::get<0>(*it_pt);
        }

        centroid_pt /= (int)(result.size() - 1);

        // reflection
        Vector<dim> reflection_pt(centroid_pt);
        reflection_pt += alpha * (centroid_pt - std::get<0>(result[result.size() - 1]));
        double reflection_score = func(reflection_pt);

        if ((std::get<1>(result[0]) <= reflection_score) && (reflection_score < std::get<1>(result[result.size() - 2])))
        {
            result.pop_back();
            result.emplace_back(reflection_pt, reflection_score);
            continue;
        }

        // expansion
        if (reflection_score < std::get<1>(result[0]))
        {
            Vector<dim> expansion_pt(centroid_pt);
            expansion_pt += gamma * (centroid_pt - std::get<0>(result[result.size() - 1]));
            double expansion_score = func(expansion_pt);

            result.pop_back();
            if (expansion_score < reflection_score)
                result.emplace_back(expansion_pt, expansion_score);
            else
                result.emplace_back(reflection_pt, reflection_score);
            continue;
        }
        // Contraction
        Vector<dim> contraction_pt(centroid_pt);
        contraction_pt += rho * (centroid_pt - std::get<0>(result[result.size() - 1]));
        double contraction_score = func(contraction_pt);
        if (contraction_score < std::get<1>(result[result.size() - 1]))
        {
            result.pop_back();
            result.emplace_back(contraction_pt, contraction_score);
            continue;
        }

        // Reduction
        auto pt_1 = std::get<0>(result[0]);
        std::vector<Tuple<dim>> reduct_result;

        for (auto it_pt = result.begin(); it_pt != result.end(); it_pt++)
        {
            Vector<dim> new_pt(pt_1);
            new_pt += sigma * (std::get<0>(*it_pt) - pt_1);
            double new_score = func(new_pt);
            reduct_result.emplace_back(new_pt, new_score);
        }

        result.clear();
        result.insert(result.end(), reduct_result.begin(), reduct_result.end());
    }

}

#endif //NELDER_MEAD_NELDERMEADOPTIMIZER_H
