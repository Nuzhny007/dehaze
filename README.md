# Dehaze Algorithm

Итерационный алгоритм для удаления дымки на изображении. <br>
Основан на вычислении количественных характеристик дымки за счёт которых происходит оптимизация карты пропускания.
Подробнее об алгоритме смотреть в ссылках.

# Реализация
Для обработки изображения использована библиотека OpenCV

# Выполнение

           Usage:
           ./dehaze <path to image> [--path_size]=<default 16> [--max_iter]=<default 10e5> [--eps]=<default 10e-7> [--lamda]=<default 4> [--tmin]=<default 0.2> [--dp]=<0.7> [--log]=<default 0> [--out]=<path to result image> [--show]=<default 1>


# Ссылки
Оригинальная статья с реализованным алгоритмом: https://www.mdpi.com/2072-4292/12/14/2233 <br>
Реализация всего алгоритма на MatLab: https://github.com/v1t0ry/OTM-AAL <br>
Реализация алгоритма Нелдера Мида для решения задачи минимизации: https://github.com/Enderdead/nelder-mead
