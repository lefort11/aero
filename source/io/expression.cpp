#include "expression.h"
#include <cctype>
#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace aero {
namespace {
// Проверяет десятичную цифру без зависимости от локали.
bool digit(char c) {
    return c >= '0' && c <= '9';
}
}

// Разбирает формулу и формирует последовательность инструкций вычисления.
class Expression::Parser {
public:
    // Связывает текст выражения с выходным массивом инструкций и начинает разбор с нуля.
    Parser(const std::string& input, std::vector<Instruction>& code) : s(input), out(code), pos(0) {}
    // Разбирает всё выражение и проверяет отсутствие лишнего текста.
    void compile() {
        if (s.size() > 65536)
            fail("formula too long");
        parse(1, 0);
        spaces();
        if (pos != s.size())
            fail("unexpected token");
    }

private:
    const std::string& s;          // Исходная формула из файла настроек.
    std::vector<Instruction>& out; // Команды в порядке вычисления.
    std::size_t pos;               // Позиция следующего символа, начиная с нуля.
    // Проверяет следующий токен выражения без изменения позиции.
    char peek() const { return pos < s.size() ? s[pos] : '\0'; }
    // Пропускает пробельные разделители перед следующим токеном.
    void spaces() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])))
            ++pos;
    }
    // Прерывает чтение или запись с диагностикой некорректных данных.
    [[noreturn]] void fail(const std::string& why) const {
        throw std::invalid_argument("formula column " + std::to_string(pos + 1) + ": " + why);
    }
    // Проверяет обязательное условие и сообщает ошибку при его нарушении.
    void require(char c) {
        spaces();
        if (peek() != c)
            fail(std::string("expected '") + c + "'");
        ++pos;
    }
    // Добавляет инструкцию в выходной код выражения.
    std::size_t emit(Op op, double v = 0) {
        out.push_back({op, v, 0});
        return out.size() - 1;
    }
    // Добавляет числовую константу в код выражения.
    void literal() {
        const std::size_t start = pos;
        while (digit(peek()))
            ++pos;
        if (peek() == '.') {
            ++pos;
            while (digit(peek()))
                ++pos;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos;
            if (peek() == '-' || peek() == '+')
                ++pos;
            if (!digit(peek()))
                fail("missing exponent");
            while (digit(peek()))
                ++pos;
        }
        std::istringstream in(s.substr(start, pos - start));
        in.imbue(std::locale::classic());
        double value;
        if (!(in >> value) || !in.eof() || !std::isfinite(value))
            fail("invalid number");
        emit(Number, value);
    }
    // Записываем переходы так, чтобы при вычислении if выполнялась только выбранная ветвь.
    void conditional(unsigned depth) {
        require('(');
        parse(1, depth + 1);
        require(',');
        const std::size_t branch = emit(JumpFalse);
        parse(1, depth + 1);
        require(',');
        const std::size_t end = emit(Jump);
        out[branch].target = out.size();
        parse(1, depth + 1);
        require(')');
        // Обе ветви оставляют на стеке ровно одно значение. Дальше исполнение общее.
        out[end].target = out.size();
    }

    // Распознаёт переменные, константу pi и вызовы функций в начальных условиях.
    void identifier(unsigned depth) {
        const std::size_t start = pos++;
        while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')
            ++pos;
        const std::string id = s.substr(start, pos - start);
        if (id == "x") {
            emit(X);
            return;
        }
        if (id == "y") {
            emit(Y);
            return;
        }
        if (id == "z") {
            emit(Z);
            return;
        }
        if (id == "t") { emit(T); return; }
        if (id == "pi") {
            emit(Number, std::acos(-1.0));
            return;
        }
        if (id == "if") {
            conditional(depth);
            return;
        }
        if (id == "e") { emit(Number, std::exp(1.)); return; }
        // Таблица задаёт имя и число аргументов для обоих входных файлов.
        struct Function { const char* name; Op op; unsigned arguments; };
        static const Function functions[] = {
            {"sin", Sin, 1},
            {"cos", Cos, 1},
            {"tan", Tan, 1},
            {"asin", Asin, 1},
            {"acos", Acos, 1},
            {"atan", Atan, 1},
            {"sinh", Sinh, 1},
            {"cosh", Cosh, 1},
            {"tanh", Tanh, 1},
            {"exp", Exp, 1},
            {"expm1", Expm1, 1},
            {"log", Log, 1},
            {"log10", Log10, 1},
            {"log2", Log2, 1},
            {"log1p", Log1p, 1},
            {"sqrt", Sqrt, 1},
            {"cbrt", Cbrt, 1},
            {"abs", Abs, 1},
            {"floor", Floor, 1},
            {"ceil", Ceil, 1},
            {"round", Round, 1},
            {"erf", Erf, 1},
            {"erfc", Erfc, 1},
            {"pow", Pow, 2},
            {"atan2", Atan2, 2},
            {"hypot", Hypot, 2},
            {"min", Min, 2},
            {"max", Max, 2},
            {"fmod", Fmod, 2},
            {"ln", Log, 1}
        };
        for (const auto& f : functions) {
            if (id != f.name) continue;
            require('(');
            parse(1, depth + 1);
            if (f.arguments == 2) {
                require(',');
                parse(1, depth + 1);
            }
            require(')');
            emit(f.op);
            return;
        }
        fail("unknown identifier " + id);
    }
    // Разбирает подвыражение до первого оператора с приоритетом ниже min_prec.
    void parse(unsigned min_prec, unsigned depth) {
        if (depth >= 64)
            fail("formula nesting exceeds 64");
        spaces();
        const char c = peek();
        if (c == '-' || c == '+' || c == '!') {
            ++pos;
            parse(6, depth + 1);
            if (c == '-')
                emit(Neg);
            if (c == '!') emit(Not);
        } else if (c == '(') {
            ++pos;
            parse(1, depth + 1);
            require(')');
        } else if (digit(c) || c == '.')
            literal();
        else if (std::isalpha(static_cast<unsigned char>(c)))
            identifier(depth);
        else
            fail("expected a number, variable or subexpression");

        // Текст оператора, его приоритет и соответствующий код вычисления.
        struct Operator {
            const char* token;
            unsigned precedence;
            Op op;
        };
        static const Operator ops[] = {{"||",1,Or}, {"&&",2,And}, {"<=",3,Le}, {">=",3,Ge},
            {"==",3,Eq}, {"!=",3,Ne}, {"<",3,Lt}, {">",3,Gt}, {"+",4,Add}, {"-",4,Sub},
            {"**",7,Pow}, {"*",5,Mul}, {"/",5,Div}, {"%",5,Fmod}, {"^",7,Pow}};
        for (;;) {
            spaces();
            std::size_t k = 0;
            for (; k < sizeof(ops) / sizeof(ops[0]); ++k) {
                if (s.compare(pos, std::char_traits<char>::length(ops[k].token), ops[k].token) == 0)
                    break;
            }
            if (k == sizeof(ops) / sizeof(ops[0]) || ops[k].precedence < min_prec)
                return;
            pos += std::char_traits<char>::length(ops[k].token);
            // При известном результате условия пропускаем правую часть.
            if (ops[k].op == And || ops[k].op == Or) {
                const bool either = ops[k].op == Or;
                if (either) emit(Not);
                const auto skip = emit(JumpFalse);
                parse(ops[k].precedence + 1, depth + 1);
                emit(Not);
                emit(Not);
                const auto end = emit(Jump);
                out[skip].target = out.size();
                emit(Number, either ? 1. : 0.);
                out[end].target = out.size();
                continue;
            }
            // Степени группируются справа (a^(b^c)), остальные операции одного уровня — слева.
            parse(ops[k].precedence + (ops[k].op == Pow ? 0u : 1u), depth + 1);
            emit(ops[k].op);
        }
    }
};
// Компилирует формулу один раз. При обходе сетки меняются только координаты узла.
Expression::Expression(const std::string& formula) {
    Parser(formula, code_).compile();
}

// Исполняет подготовленную последовательность операций над локальным стеком значений.
double Expression::evaluate(double x, double y, double z, double t) const {
    // Каждый ожидающий правый операнд занимает один уровень рекурсивного разбора.
    double stack[65];
    std::size_t size = 0;
    for (std::size_t pc = 0; pc < code_.size();) {
        const Instruction& i = code_[pc++];
        double v = 0;
        switch (i.op) {
        case Number:
            v = i.value;
            break;
        case X:
            v = x;
            break;
        case Y:
            v = y;
            break;
        case Z:
            v = z;
            break;
        case T:
            v = t;
            break;
        case JumpFalse:
            if (!stack[--size])
                pc = i.target;
            continue;
        case Jump:
            pc = i.target;
            continue;
        case Neg:
            v = -stack[--size];
            break;
        case Not: v = !stack[--size]; break;
        case Sin: v = std::sin(stack[--size]); break;
        case Cos: v = std::cos(stack[--size]); break;
        case Tan: v = std::tan(stack[--size]); break;
        case Asin: v = std::asin(stack[--size]); break;
        case Acos: v = std::acos(stack[--size]); break;
        case Atan: v = std::atan(stack[--size]); break;
        case Sinh: v = std::sinh(stack[--size]); break;
        case Cosh: v = std::cosh(stack[--size]); break;
        case Tanh: v = std::tanh(stack[--size]); break;
        case Exp: v = std::exp(stack[--size]); break;
        case Expm1: v = std::expm1(stack[--size]); break;
        case Log: v = std::log(stack[--size]); break;
        case Log10: v = std::log10(stack[--size]); break;
        case Log2: v = std::log2(stack[--size]); break;
        case Log1p: v = std::log1p(stack[--size]); break;
        case Sqrt: v = std::sqrt(stack[--size]); break;
        case Cbrt: v = std::cbrt(stack[--size]); break;
        case Abs: v = std::abs(stack[--size]); break;
        case Floor: v = std::floor(stack[--size]); break;
        case Ceil: v = std::ceil(stack[--size]); break;
        case Round: v = std::round(stack[--size]); break;
        case Erf: v = std::erf(stack[--size]); break;
        case Erfc: v = std::erfc(stack[--size]); break;
        default: {
            // Правый аргумент положили в стек последним. Порядок важен для вычитания и деления.
            const double b = stack[--size], a = stack[--size];
            switch (i.op) {
            case Add:
                v = a + b;
                break;
            case Sub:
                v = a - b;
                break;
            case Mul:
                v = a * b;
                break;
            case Div:
                if (b == 0)
                    throw std::domain_error("division by zero");
                v = a / b;
                break;
            case Pow:
                v = std::pow(a, b);
                break;
            case Atan2: v = std::atan2(a, b); break;
            case Hypot: v = std::hypot(a, b); break;
            case Min: v = std::fmin(a, b); break;
            case Max: v = std::fmax(a, b); break;
            case Fmod: v = std::fmod(a, b); break;
            case Lt:
                v = a < b;
                break;
            case Le:
                v = a <= b;
                break;
            case Gt:
                v = a > b;
                break;
            case Ge:
                v = a >= b;
                break;
            case Eq:
                v = a == b;
                break;
            case Ne:
                v = a != b;
                break;
            default:
                throw std::logic_error("invalid expression opcode");
            }
        }
        }
        if (!std::isfinite(v))
            throw std::domain_error("non-finite expression result");
        stack[size++] = v;
    }
    if (size != 1)
        throw std::logic_error("empty or invalid expression");
    return stack[0];
}
}
