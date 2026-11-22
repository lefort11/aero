#pragma once
#include "grad.h"
#include <algorithm>

namespace aero {
namespace viscosity {
// Перечисляет тетраэдры, по которым AES усредняет вязкий оператор объёмного элемента.
// Первым аргументом visit получает вес разбиения. Порядок вершин один и тот же
// для таблицы CPU, вычисления на GPU и проверки якобиана.
template<class Visit>
bool for_volume_tetrahedra(const Element& e,Visit visit) {
    if (e.count==4) {
        visit(1.,0,1,2,3);
        return true;
    }
    if (e.count==8)
        return for_hex_tetrahedra(e,[&](unsigned a,unsigned b,unsigned c,unsigned d) {
            visit(.25,a,b,c,d);
        });
    if (e.count==5) {
        // Единственная вершина пирамиды смежна со всеми четырьмя узлами основания.
        for (unsigned apex=0;apex<5;++apex) if (e.neighbours[apex]==(31u^(1u<<apex))) {
            for (unsigned skip=0;skip<5;++skip) if (skip!=apex) {
                unsigned base[3],n=0;
                for (unsigned j=0;j<5;++j) if (j!=skip && j!=apex) base[n++]=j;
                visit(.5,base[0],base[1],base[2],apex);
            }
            return true;
        }
        return false;
    }
    if (e.count!=6) return false;
    // Находим треугольное основание призмы и соответствие с верхним основанием.
    for (unsigned a=0;a<6;++a) for (unsigned b=a+1;b<6;++b) for (unsigned c=b+1;c<6;++c) {
        if (!(e.edge(a,b) && e.edge(b,c) && e.edge(c,a))) continue;
        unsigned bottom[]={a,b,c},top[6]{};
        for (unsigned i:bottom) {
            unsigned matches=0;
            for (unsigned j=0;j<6;++j)
                if (j!=a && j!=b && j!=c && e.edge(i,j)) { top[i]=j; ++matches; }
            if (matches!=1) return false;
        }
        if (!(e.edge(top[a],top[b]) && e.edge(top[b],top[c]) && e.edge(top[c],top[a]))) return false;
        // Каждая перестановка даёт три тетраэдра. По шести разбиениям берём среднее.
        do {
            const unsigned i=bottom[0],j=bottom[1],k=bottom[2];
            visit(1./6.,i,j,k,top[k]);
            visit(1./6.,i,j,top[j],top[k]);
            visit(1./6.,i,top[i],top[j],top[k]);
        } while (std::next_permutation(bottom,bottom+3));
        return true;
    }
    return false;
}
}
}
