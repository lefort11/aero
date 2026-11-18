#ifndef AERO_IO_EXPRESSION_H
#define AERO_IO_EXPRESSION_H
#include <string>
#include <vector>

namespace aero {
// Immutable after construction. evaluate() uses only a local operand stack.
class Expression {
public:
    explicit Expression(const std::string& formula);
    double evaluate(double x,double y,double z) const;
private:
    enum Op { Number,X,Y,Z,Neg,Add,Sub,Mul,Div,Pow,Lt,Le,Gt,Ge,Eq,Ne,
              Sin,Cos,Exp,Sqrt,Abs,JumpFalse,Jump };
    struct Instruction { Op op; double value; std::size_t target; };
    class Parser;
    std::vector<Instruction> code_;
};
}
#endif
