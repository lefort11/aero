#ifndef AERO_IO_EXPRESSION_H
#define AERO_IO_EXPRESSION_H
#include <string>
#include <vector>

namespace aero {
// Конструктор разбирает формулу один раз. При вычислении каждый вызов пользуется собственным стеком операндов.
class Expression {
public:
    explicit Expression(const std::string& formula);
    // Подставляет координаты текущего узла и возвращает одно конечное число.
    double evaluate(double x, double y, double z) const;

private:
    enum Op {
        Number,
        X,
        Y,
        Z,
        Neg,
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Lt,
        Le,
        Gt,
        Ge,
        Eq,
        Ne,
        Sin,
        Cos,
        Exp,
        Sqrt,
        Abs,
        JumpFalse,
        Jump
    };
    // Одна инструкция для стекового вычислителя: операция, числовой литерал или цель условного перехода.
    struct Instruction {
        Op op;
        double value;
        // Номер следующей инструкции при выполнении перехода.
        std::size_t target;
    };
    class Parser;
    // Готовая программа выражения. Узлы сетки читают её параллельно.
    std::vector<Instruction> code_;
};
}
#endif
