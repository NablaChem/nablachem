#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>

inline double
log_binomial(int n, int k)
{
    return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
}

double log_permutation_factor(const std::vector<int> counts, const int total)
{
    int logscore = 0;
    int remaining = total;
    for (const auto count : counts)
    {
        logscore += log_binomial(remaining, count);
        remaining -= count;
    }
    return logscore;
}

double
pure_prediction(const std::vector<int> &label, double a, double b, double log_puresize)
{
    int M = 0;
    double logscore = 0;

    int last_d = label[0];
    std::vector<int> counts;
    int degree, count, total;
    total = 0;
    for (long unsigned int i = 0; i < label.size() / 2; i++)
    {
        degree = label[i * 2];
        count = label[i * 2 + 1];
        if (degree != last_d)
        {
            logscore += log_permutation_factor(counts, total);
            M += total * last_d;
            last_d = degree;
            total = 0;
            counts.clear();
        }
        counts.push_back(count);
        total += count;
    }
    logscore += log_permutation_factor(counts, total);
    M += total * last_d;

    double prefactor = logscore / M + 1;
    double lgdu = (log_puresize - b) / a;
    return exp(a * prefactor * lgdu + b);
}

bool is_pure(const std::vector<int> &label)
{
    int lastdeg = -1;
    int degree;
    for (long unsigned int i = 0; i < label.size() / 2; i++)
    {
        degree = label[i * 2];
        if (degree == lastdeg)
        {
            return false;
        }
        lastdeg = degree;
    }
    return true;
}

PYBIND11_MODULE(space, m)
{
    m.doc() = "nablachem hotspots implemented in C++";

    m.def("pure_prediction", &pure_prediction, "Implements the prefactor in ApproximateCounter._pure_prediction.");
    m.def("is_pure", &is_pure, "Tests whether a label is from a pure degree sequence.");
}