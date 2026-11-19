#include "cli.h"
#include <iostream>

// Передаёт параметры импортёру и возвращает его код завершения оболочке.
int main(int argc, char** argv) {
    return aero::converter::run_cli(argc, argv, std::cout, std::cerr);
}
