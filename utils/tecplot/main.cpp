#include "tecplot.h"
#include <iostream>
// Командная строка экспортёра использует тот же разбор параметров, что и его тесты.
int main(int argc, char** argv) {
    return aero::tecplot::run_cli(argc, argv, std::cout, std::cerr);
}
