#include "cli.h"
#include <iostream>

int main(int argc, char** argv) {
    return aero::converter::run_cli(argc, argv, std::cout, std::cerr);
}
