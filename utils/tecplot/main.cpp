#include "tecplot.h"
#include <iostream>
int main(int argc,char** argv) {return aero::tecplot::run_cli(argc,argv,std::cout,std::cerr);}
