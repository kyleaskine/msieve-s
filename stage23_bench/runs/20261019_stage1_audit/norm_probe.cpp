#include <iostream>
#include <iomanip>
#include "norms.h"
int main() {
    s23::DPoly p; p.deg=5;
    double skew;
    while(std::cin >> skew) {
        for(int i=0;i<6;i++) if(!(std::cin >> p.c[i])) return 2;
        double s = skew > 0 ? skew : s23::L2_skewness_d(p);
        std::cout << std::setprecision(17) << s << ' ' << s23::L2_lognorm_d(p,s) << '\n';
    }
}
