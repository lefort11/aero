#ifndef AERO_IO_EXPRESSION_H
#define AERO_IO_EXPRESSION_H
#include <string>
#include <vector>

namespace aero {
// Конструктор разбирает формулу один раз. При вычислении каждый вызов пользуется собственным стеком операндов.
class Expression {
public:
    // Компилирует формулу один раз. При обходе сетки меняются только координаты узла.
    explicit Expression(const std::string& formula);
    // Подставляет координаты текущего узла и возвращает одно конечное число.
    double evaluate(double x, double y, double z, double t = 0.) const;

private:
    // Коды инструкций интерпретатора математических формул.
    enum Op {
        Number,
        X,
        Y,
        Z,
        T,
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
        Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Expm1, Log, Log10, Log2, Log1p, Cbrt, Floor, Ceil, Round, Erf, Erfc, Atan2, Hypot, Min, Max, Fmod,
        Not, And, Or,
        JumpFalse,
        Jump
    };
    // Одна инструкция для стекового вычислителя: операция, числовой литерал или цель условного перехода.
    struct Instruction {
        // Операция, выполняемая этой инструкцией.
        Op op;
        // Числовой операнд инструкции.
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
