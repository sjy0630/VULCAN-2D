// Numerical load-line helper. Physical definitions and validation are in
// bidirectional_state.py. No measured memristor current enters this solver.
#include <cmath>
#include <algorithm>

namespace {
double interp(double x, const double* vx, const double* iy, int n) {
    int low = 0, high = n - 1;
    if (x <= vx[0]) return iy[0];
    if (x >= vx[high]) return iy[high];
    while (high - low > 1) {
        int mid = (low + high) / 2;
        if (vx[mid] <= x) low = mid; else high = mid;
    }
    return iy[low] + (iy[high]-iy[low])*(x-vx[low])/(vx[high]-vx[low]);
}
double logsinh(double x) { return x + std::log(-std::expm1(-2*x)) - std::log(2.); }
double logadd(double x, double y) {
    double high = std::max(x, y), low = std::min(x, y);
    return high + std::log1p(std::exp(low-high));
}
double logcurrent(double v, double lp, const double* p) {
    if (v == 0.) return -INFINITY;
    double h = std::log(10.)*p[0] + std::log(v/.2) + p[1]*(std::sqrt(v)-std::sqrt(.2));
    double c = std::log(10.)*p[2] + logsinh(p[3]*v)-logsinh(p[3]*.2) + lp;
    return logadd(h, c);
}
double internal(double applied, double lp, const double* p, const double* vx, const double* iy, int n) {
    if (applied == 0.) return 0.;
    double low = 0., high = applied;
    for (int k = 0; k < 48; ++k) {
        double mid = (low+high)/2.;
        double it = interp(applied-mid, vx, iy, n);
        if (logcurrent(mid, lp, p) < std::log(it)) low = mid; else high = mid;
    }
    return (low+high)/2.;
}
}

extern "C" {
void bd_load(double applied, double lp, const double* p, const double* vx, const double* iy, int n, double* result) {
    double vh = internal(applied, lp, p, vx, iy, n);
    result[0] = vh;
    result[1] = applied == 0. ? 0. : std::exp(logcurrent(vh, lp, p));
}
double bd_rhs(double applied, double lp, const double* p, double vt, double ell,
              const double* vx, const double* iy, int n) {
    double vh = internal(applied, lp, p, vx, iy, n);
    double common = ell+p[8]*vh*vh/vt;
    double build = common-p[5]/(2*vt)+p[7]*p[6]*vh/vt;
    double remove = common+p[5]/(2*vt)-(1-p[7])*p[6]*vh/vt;
    // d log(s)/dx = k_build*(1/s - 1) - k_remove. expm1
    // preserves the near-s=1 limit. LSODA may probe slightly above 1.
    return std::exp(build)*std::expm1(-lp)-std::exp(remove);
}
}
