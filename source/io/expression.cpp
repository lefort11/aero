#include "expression.h"
#include <cctype>
#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace aero {
namespace {
bool digit(char c) { return c>='0' && c<='9'; }
}

class Expression::Parser {
public:
    Parser(const std::string& input,std::vector<Instruction>& code):s(input),out(code),pos(0) {}
    void compile() {
        if (s.size()>65536) fail("formula too long");
        parse(1,0);
        spaces();
        if (pos!=s.size()) fail("unexpected token");
    }
private:
    const std::string& s;
    std::vector<Instruction>& out;
    std::size_t pos;
    char peek() const { return pos<s.size()?s[pos]:'\0'; }
    void spaces() { while (pos<s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos; }
    [[noreturn]] void fail(const std::string& why) const {
        throw std::invalid_argument("formula column "+std::to_string(pos+1)+": "+why);
    }
    void require(char c) {
        spaces();
        if (peek()!=c) fail(std::string("expected '")+c+"'");
        ++pos;
    }
    std::size_t emit(Op op,double v=0) {
        out.push_back({op,v,0});
        return out.size()-1;
    }
    void literal() {
        const std::size_t start=pos;
        while (digit(peek())) ++pos;
        if (peek()=='.') { ++pos; while (digit(peek())) ++pos; }
        if (peek()=='e' || peek()=='E') {
            ++pos;
            if (peek()=='-' || peek()=='+') ++pos;
            if (!digit(peek())) fail("missing exponent");
            while (digit(peek())) ++pos;
        }
        std::istringstream in(s.substr(start,pos-start));
        in.imbue(std::locale::classic());
        double value;
        if (!(in>>value) || !in.eof() || !std::isfinite(value)) fail("invalid number");
        emit(Number,value);
    }
    void identifier(unsigned depth) {
        const std::size_t start=pos++;
        while (std::isalnum(static_cast<unsigned char>(peek())) || peek()=='_') ++pos;
        const std::string id=s.substr(start,pos-start);
        if (id=="x") { emit(X); return; }
        if (id=="y") { emit(Y); return; }
        if (id=="z") { emit(Z); return; }
        if (id=="pi") { emit(Number,std::acos(-1.0)); return; }
        if (id=="if") {
            require('('); parse(1,depth+1); require(',');
            const std::size_t branch=emit(JumpFalse);
            parse(1,depth+1); require(',');
            const std::size_t end=emit(Jump);
            out[branch].target=out.size();
            parse(1,depth+1); require(')');
            out[end].target=out.size();
            return;
        }
        Op op;
        if (id=="sin") op=Sin;
        else if (id=="cos") op=Cos;
        else if (id=="sqrt") op=Sqrt;
        else if (id=="exp") op=Exp;
        else if (id=="abs") op=Abs;
        else fail("unknown identifier "+id);
        require('('); parse(1,depth+1); require(')'); emit(op);
    }
    void parse(unsigned min_prec,unsigned depth) {
        if (depth>=64) fail("formula nesting exceeds 64");
        spaces();
        const char c=peek();
        if (c=='-' || c=='+') {
            ++pos; parse(4,depth+1); if (c=='-') emit(Neg);
        } else if (c=='(') {
            ++pos; parse(1,depth+1); require(')');
        } else if (digit(c) || c=='.') literal();
        else if (std::isalpha(static_cast<unsigned char>(c))) identifier(depth);
        else fail("expected a number, variable or subexpression");

        struct Operator { const char* token; unsigned precedence; Op op; };
        static const Operator ops[]={{"<=",1,Le},{">=",1,Ge},{"==",1,Eq},{"!=",1,Ne},
            {"<",1,Lt},{">",1,Gt},{"+",2,Add},{"-",2,Sub},{"*",3,Mul},{"/",3,Div},{"^",5,Pow}};
        for (;;) {
            spaces();
            std::size_t k=0;
            for (;k<sizeof(ops)/sizeof(ops[0]);++k) {
                if (s.compare(pos,std::char_traits<char>::length(ops[k].token),ops[k].token)==0) break;
            }
            if (k==sizeof(ops)/sizeof(ops[0]) || ops[k].precedence<min_prec) return;
            pos+=std::char_traits<char>::length(ops[k].token);
            parse(ops[k].precedence+(ops[k].op==Pow?0u:1u),depth+1);
            emit(ops[k].op);
        }
    }
};
Expression::Expression(const std::string& formula) { Parser(formula,code_).compile(); }

double Expression::evaluate(double x,double y,double z) const {
    // Each pending right operand consumes a level of the bounded parse recursion.
    double stack[65];
    std::size_t size=0;
    for (std::size_t pc=0;pc<code_.size();) {
        const Instruction& i=code_[pc++];
        double v=0;
        switch(i.op) {
        case Number: v=i.value; break;
        case X: v=x; break;
        case Y: v=y; break;
        case Z: v=z; break;
        case JumpFalse: if (!stack[--size]) pc=i.target; continue;
        case Jump: pc=i.target; continue;
        case Neg: v=-stack[--size]; break;
        case Sin: v=std::sin(stack[--size]); break;
        case Cos: v=std::cos(stack[--size]); break;
        case Sqrt: v=std::sqrt(stack[--size]); break;
        case Exp: v=std::exp(stack[--size]); break;
        case Abs: v=std::fabs(stack[--size]); break;
        default: {
            const double b=stack[--size],a=stack[--size];
            switch(i.op) {
            case Add: v=a+b; break;
            case Sub: v=a-b; break;
            case Mul: v=a*b; break;
            case Div:
                if (b==0) throw std::domain_error("division by zero");
                v=a/b; break;
            case Pow: v=std::pow(a,b); break;
            case Lt: v=a<b; break;
            case Le: v=a<=b; break;
            case Gt: v=a>b; break;
            case Ge: v=a>=b; break;
            case Eq: v=a==b; break;
            case Ne: v=a!=b; break;
            default: throw std::logic_error("invalid expression opcode");
            }
        }}
        if (!std::isfinite(v)) throw std::domain_error("non-finite expression result");
        stack[size++]=v;
    }
    if (size!=1) throw std::logic_error("empty or invalid expression");
    return stack[0];
}
}
