#pragma once
#include "../parallel/fixed_matrix.h"
#include "../mesh/coordinate.h"
#include "model_math.h"

namespace aero {
namespace turbulence {
enum class Scale { Off, Max, Min, Vol, Omega, OmegaTilde, Sla };
inline const char* scale_name(Scale scale) {
    switch (scale) {
    case Scale::Max: return "max";
    case Scale::Min: return "min";
    case Scale::Vol: return "vol";
    case Scale::Omega: return "delta_omega";
    case Scale::OmegaTilde: return "delta_omega_tilde";
    case Scale::Sla: return "sla";
    default: return "off";
    }
}
struct LesSettings {
    Scale scale = Scale::Off;
    // Для трёх направленных масштабов размер фильтра зависит от текущей скорости.
    AERO_HOST_DEVICE bool directional() const {
        return scale==Scale::Omega || scale==Scale::OmegaTilde || scale==Scale::Sla;
    }
    bool switch_on_vtm = true; // В SLA учитываем наклон вихря относительно тензора деформаций.
    double f_kh_lim = .1;      // Нижний предел множителя F_KH.
    double c_sigma = 1.35;     // Константа модели Никуд.
};
using VelocityGradient = FixedMatrix<double,3,3>; // Строка i содержит производные компоненты скорости u_i.

// Повороты Якоби ортогонализуют столбцы градиента, сохраняя сингулярные числа.
// Длины итоговых столбцов дают их непосредственно, в том числе у почти плоского течения.
AERO_HOST_DEVICE inline double jacobi_sigma_rate(const VelocityGradient& g) {
    Coordinate col[3];
    for (unsigned j=0;j<3;++j) for (unsigned i=0;i<3;++i) col[j][i]=g(i, j);
    for (unsigned sweep=0;sweep<12;++sweep) {
        bool rotated=false;
        for (unsigned i=0;i<2;++i) for (unsigned j=i+1;j<3;++j) {
            const double p=dot3(col[i],col[i]), q=dot3(col[j],col[j]), r=dot3(col[i],col[j]);
            if (::fabs(r)<=1.e-15*::sqrt(p)*::sqrt(q)) continue;
            const double gap=.5*(q-p);
            const double t=r/(gap+::copysign(::hypot(gap,r),gap));
            const double c=1./::sqrt(1.+t*t), s=t*c;
            for (unsigned k=0;k<3;++k) {
                const double a=col[i][k], b=col[j][k];
                col[i][k]=c*a-s*b;
                col[j][k]=s*a+c*b;
            }
            rotated=true;
        }
        if (!rotated) break;
    }
    const double a=::sqrt(dot3(col[0],col[0])), b=::sqrt(dot3(col[1],col[1])), c=::sqrt(dot3(col[2],col[2]));
    const double hi=::fmax(::fmax(a,b),c), lo=::fmin(::fmin(a,b),c);
    const double mid=::fmax(::fmin(a,b),::fmin(::fmax(a,b),c));
    return hi>0. ? lo*((hi-mid)/hi)*((mid-lo)/hi) : 0.;
}

// Для старшего корня кубического уравнения нужен cos(acos(r)/3).
// Подстановка cos(acos(r)/3) = 1/2 + t*p, t = sqrt((1+r)/2),
// даёт p²(3+2tp)=1. На всём интервале p меняется от 1/sqrt(3) до 1/2.
// Начинаем с прямой между этими значениями. Две поправки Галлея
// восстанавливают корень с точностью double, в том числе возле концов интервала.
AERO_HOST_DEVICE inline double largest_root_cosine(double r) {
    const double t=::sqrt(.5*(1.+::fmin(1.,::fmax(-1.,r))));
    const double at_zero=.57735026918962576451;
    double p=at_zero+t*(.5-at_zero);
    for (unsigned iteration=0;iteration<2;++iteration) {
        const double tp=t*p;
        const double residual=p*p*(3.+2.*tp)-1.;
        const double slope=6.*p*(1.+tp);
        const double half_curvature=3.+6.*tp;
        p-=residual*slope*reciprocal(slope*slope-residual*half_curvature);
    }
    return .5+t*p;
}

// Быстрый путь для градиента безопасного масштаба. Наибольшее сингулярное число
// находим из характеристического уравнения симметричной матрицы G^T G.
// Два малых числа восстанавливаем по минорам и определителю: разность следа
// и наибольшего корня теряла бы значащие цифры у почти плоского течения.
AERO_HOST_DEVICE inline double sigma_rate(const VelocityGradient& g) {
    const Coordinate a{{g(0, 0),g(1, 0),g(2, 0)}}, b{{g(0, 1),g(1, 1),g(2, 1)}}, c{{g(0, 2),g(1, 2),g(2, 2)}};
    const double aa=dot3(a,a), bb=dot3(b,b), cc=dot3(c,c);
    const double ab=dot3(a,b), ac=dot3(a,c), bc=dot3(b,c);
    const double mean=(aa+bb+cc)/3.;
    const double x=aa-mean, y=bb-mean, z=cc-mean;
    const double spread=::sqrt((x*x+y*y+z*z+2.*(ab*ab+ac*ac+bc*bc))/6.);
    if (spread==0.) return 0.;
    if (spread<1.e-8*mean) return jacobi_sigma_rate(g);
    const double determinant=x*(y*z-bc*bc)+ab*(2.*ac*bc-ab*z)-y*ac*ac;
    const double cosine=determinant*reciprocal(2.*spread*spread*spread);
    // Вблизи кратного верхнего корня относительная точность разности σ1−σ2
    // важнее стоимости редкого перехода к ортогонализации.
    if (cosine < -.999999) return jacobi_sigma_rate(g);
    const double high2=mean+2.*spread*largest_root_cosine(cosine);
    const double high=::sqrt(high2);
    const Coordinate bc_minor=cross3(b,c);
    const double inverse_high=reciprocal(high);
    const double product=::fabs(dot3(a,bc_minor))*inverse_high; // σ2 σ3
    if (product==0.) return 0.;
    // След GᵀG даёт сумму квадратов всех трёх сингулярных чисел.
    // Вычитаем старшее число только при хорошем численном разделении.
    // У почти одномерного течения или близких малых чисел возвращаемся к минорам.
    const double trace=aa+bb+cc;
    double sum2=trace-high2;
    if (!(sum2>.125*trace && sum2-2.*product>.0625*sum2)) {
        const Coordinate ab_minor=cross3(a,b), ac_minor=cross3(a,c);
        const double minors=dot3(ab_minor,ab_minor)+dot3(ac_minor,ac_minor)+dot3(bc_minor,bc_minor);
        sum2=(minors-product*product)*reciprocal(high2);
    }
    if (sum2<=0.) return 0.;
    const double gap2=sum2-2.*product;
    if (gap2<1.e-8*sum2) return jacobi_sigma_rate(g);
    const double gap=::sqrt(gap2);
    const double mid=.5*(::sqrt(sum2+2.*product)+gap);
    return (product*reciprocal(mid))*((high-mid)*inverse_high)*(gap*inverse_high);
}

// Квадрат длины пары вершин в поперечном масштабе Δ̃ω. Ось завихренности
// в модели единичная. Формула сохраняет исходное масштабирование и для произвольной оси.
AERO_HOST_DEVICE inline double transverse_square(const Coordinate& axis, const Coordinate& diagonal) {
    const Coordinate projection=cross3(axis,diagonal);
    return dot3(projection,projection);
}

// Четыре длиннейшие пары часто определяют весь диаметр. Геометрия задаёт только
// способ чтения пары: по номерам вершин или из таблицы готовых диагоналей.
template<class Geometry>
AERO_HOST_DEVICE inline bool try_transverse_filter(const Geometry& cell, const Coordinate& axis,
                                                    double remaining_length2, double& filter2) {
    double diameter2=0.;
    for (unsigned k=0; k<4; ++k)
        diameter2=::fmax(diameter2,transverse_square(axis,cell.long_diagonal(k)));

    // |axis × d|² ≤ |axis|² |d|². Если даже верхняя граница для пятой пары
    // меньше найденного максимума, все остальные пары можно пропустить.
    // Возле равенства или при потере нормальной точности проверяем все вершины.
    const double upper=remaining_length2*dot3(axis,axis);
    if (!(upper>=DBL_MIN && diameter2>upper*(1.+32.*DBL_EPSILON))) return false;
    filter2=diameter2/3.;
    return true;
}

// Для масштаба фильтра нужны только размер и координаты элемента. В упакованной
// таблице LES производные уже записаны по отдельности, рядом для соседних элементов.
struct FilterLengths {
    double volume, vol_length, min_edge, max_edge;
    double gradient_sum; // Максимальная сумма модулей производных по одному направлению.
    double remaining_length2;
    unsigned char longest_pairs[4];
    unsigned diagonals;
    bool rectangular_box, bounded_diameter;
    // Пара хранится как 8*i+j. Координаты могут лежать в таблице формы или в общей сетке.
    template<class Points>
    AERO_HOST_DEVICE Coordinate diagonal(const Points& points,unsigned k) const {
        const unsigned pair=longest_pairs[k];
        return points.point(pair/8)-points.point(pair%8);
    }
    // Для несжатого элемента полный перебор даёт точный максимум при любом
    // направлении оси, в том числе если максимум задаёт короткое ребро.
    template<class Points>
    AERO_HOST_DEVICE double transverse(const Points& points,unsigned count, const Coordinate& axis) const {
        double diameter2=0.;
        for (unsigned i=1; i<count; ++i) for (unsigned j=0; j<i; ++j)
            diameter2=::fmax(diameter2,transverse_square(axis,points.point(i)-points.point(j)));
        return diameter2/3.;
    }
    template<class Points>
    AERO_HOST_DEVICE double squared(const Points& points,Scale scale, unsigned count, const Coordinate& axis) const {
        if (scale==Scale::Max) return max_edge*max_edge;
        if (scale==Scale::Min) return min_edge*min_edge;
        if (scale==Scale::Vol || dot3(axis,axis)==0.) return vol_length*vol_length;
        if (rectangular_box) {
            const Coordinate diagonal=points.point(0);
            const double x=::fabs(diagonal[0]*axis[0]), y=::fabs(diagonal[1]*axis[1]);
            const double z=::fabs(diagonal[2]*axis[2]), span=x+y+z;
            if (scale==Scale::Omega) return span>0. ? volume/span : vol_length*vol_length;
            // Из четырёх диагоналей максимальную проекцию поперёк оси имеет та,
            // у которой минимален модуль продольной проекции |±x±y±z|.
            const double nearest=2.*::fmax(x,::fmax(y,z))-span;
            const double full=dot3(diagonal,diagonal)*dot3(axis,axis);
            const double transverse=full-nearest*nearest;
            // Для тонкого элемента разность квадратов может терять точность.
            // Тогда ниже считаем проекции непосредственно векторным произведением.
            if (transverse>.25*full && full<DBL_MAX) return transverse/3.;
        }
        if (diagonals) {
            double span=0.;
            for (unsigned i=0;i<diagonals;++i) {
                double value;
                if (scale==Scale::Omega) value=::fabs(dot3(points.point(i),axis));
                else value=transverse_square(axis,points.point(i));
                if (value>span) span=value;
            }
            if (scale!=Scale::Omega) return span/3.;
            return span>0. ? volume/span : vol_length*vol_length;
        }
        if (scale==Scale::Omega) {
            // Среднее сечение поперёк вихря равно объёму, делённому на толщину вдоль оси.
            double low=dot3(points.point(0),axis), high=low;
            for (unsigned i=1;i<count;++i) {
                const double x=dot3(points.point(i),axis);
                low=::fmin(low,x); high=::fmax(high,x);
            }
            return high>low ? volume/(high-low) : vol_length*vol_length;
        }
        double filter2;
        if (bounded_diameter && try_transverse_filter(points,axis,remaining_length2,filter2)) return filter2;
        return transverse(points,count,axis);
    }
};
struct FilterExtent : FilterLengths {
    Coordinate position[8];
    AERO_HOST_DEVICE const Coordinate& point(unsigned i) const { return position[i]; }
    AERO_HOST_DEVICE Coordinate long_diagonal(unsigned k) const { return diagonal(*this,k); }
    AERO_HOST_DEVICE double full_transverse_filter(unsigned count,const Coordinate& axis) const {
        return transverse(*this,count,axis);
    }
    AERO_HOST_DEVICE double length_squared(Scale scale,unsigned count,const Coordinate& axis) const {
        return squared(*this,scale,count,axis);
    }
};
struct FilterCell : FilterExtent {
    Coordinate gradient[8];           // Средние производные базисных функций в вершинах.
    Coordinate face_gradient;         // Коэффициенты средней производной для прямоугольного элемента.
    unsigned char corner_node[8];     // Соответствие локальной вершины геометрическому углу.
    AERO_HOST_DEVICE double basis(unsigned vertex,unsigned direction) const {
        return gradient[vertex][direction];
    }
    // Короткая пара не может дать проекцию длиннее неё самой. Во многих
    // искажённых гексаэдрах достаточно проверить четыре пространственные
    // диагонали. Пятую длину храним для строгой проверки во время расчёта.
    void prepare_diameter_bound(unsigned count) {
        bounded_diameter=count>=5;
        if (!bounded_diameter) return;
        double length[5]={-1.,-1.,-1.,-1.,-1.};
        unsigned char pair[5]{};
        for (unsigned i=1;i<count;++i) for (unsigned j=0;j<i;++j) {
            const Coordinate d=position[i]-position[j];
            const double square=dot3(d,d);
            for (unsigned k=0;k<5;++k) if (square>length[k]) {
                for (unsigned l=4;l>k;--l) { length[l]=length[l-1]; pair[l]=pair[l-1]; }
                length[k]=square; pair[k]=static_cast<unsigned char>(8*i+j);
                break;
            }
        }
        for (unsigned i=0;i<4;++i) longest_pairs[i]=pair[i];
        remaining_length2=length[4];
    }
    // В проекции центр симметрии сохраняется. Диаметр равен удвоенному
    // максимальному расстоянию от центра: на гексаэдре достаточно четырёх диагоналей.
    // Сжатие разрешено только при точном попарном совпадении исходной геометрии.
    void compress_symmetric_vertices(unsigned count) {
        if (count%2) return;
        Coordinate center_sum{}, lower{}, upper{};
        for (unsigned k=0;k<3;++k) {
            double low=position[0][k], high=low;
            for (unsigned i=1;i<count;++i) {
                low=::fmin(low,position[i][k]); high=::fmax(high,position[i][k]);
            }
            lower[k]=low; upper[k]=high; center_sum[k]=low+high;
        }
        unsigned corners=0;
        if (count==8) for (unsigned i=0;i<count;++i) {
            unsigned corner=0;
            for (unsigned k=0;k<3;++k) {
                if (position[i][k]==upper[k]) corner|=1u<<k;
                else if (position[i][k]!=lower[k]) { corner=8; break; }
            }
            corners|=1u<<corner;
            if (corner<8) corner_node[corner]=static_cast<unsigned char>(i);
        }
        bool used[8]{};
        Coordinate differences[4];
        unsigned pairs=0;
        for (unsigned i=0;i<count;++i) {
            if (used[i]) continue;
            unsigned j=i+1;
            for (;j<count;++j) {
                const Coordinate sum=position[i]+position[j];
                if (!used[j] && sum[0]==center_sum[0] && sum[1]==center_sum[1] && sum[2]==center_sum[2]) break;
            }
            if (j==count) return;
            used[j]=true;
            longest_pairs[pairs]=static_cast<unsigned char>(8*i+j);
            differences[pairs++]=position[i]-position[j];
        }
        for (unsigned i=0;i<pairs;++i) position[i]=differences[i];
        diagonals=pairs;
        rectangular_box=corners==255;
        if (rectangular_box) for (unsigned k=0;k<3;++k) face_gradient[k]=.25/(upper[k]-lower[k]);
    }
    // Для асимметричной формы сохраняем границу на остальные длины только
    // тогда, когда модель действительно запрашивает поперечный диаметр.
    void prepare_geometry(unsigned count,bool transverse=true) {
        compress_symmetric_vertices(count);
        if (!diagonals && transverse) prepare_diameter_bound(count);
    }
    // Средний градиент равен разности средних скоростей на противоположных
    // гранях, делённой на расстояние между ними. Четыре пространственные
    // диагонали дают те же три разности и точно убирают постоянную скорость.
    template<class Node,class Gas>
    AERO_HOST_DEVICE double rectangular_gradient(const Node* nodes,size_t stride,const Gas* gas,
                                                   bool need_speed,VelocityGradient& gradient_out) const {
        FixedMatrix<double,4,3> jumps;
        double speed=0.;
        for (unsigned k=0;k<4;++k) {
            const auto& low=gas[nodes[k*stride]];
            const auto& high=gas[nodes[(7-k)*stride]];
            for (unsigned i=0;i<3;++i) {
                jumps(k, i)=high.velocity[i]-low.velocity[i];
                if (need_speed) {
                    const double a=::fabs(low.velocity[i]), b=::fabs(high.velocity[i]);
                    if (a>speed) speed=a;
                    if (b>speed) speed=b;
                }
            }
        }
        for (unsigned i=0;i<3;++i) {
            const double lower=jumps(0, i)+jumps(1, i), upper=jumps(2, i)+jumps(3, i);
            gradient_out(i, 0)=((jumps(0, i)-jumps(1, i))+(jumps(2, i)-jumps(3, i)))*face_gradient[0];
            gradient_out(i, 1)=(lower-upper)*face_gradient[1];
            gradient_out(i, 2)=(lower+upper)*face_gradient[2];
        }
        return speed;
    }
};
AERO_HOST_DEVICE inline double vortex_tilting(const VelocityGradient& g, const Coordinate& axis) {
    const double trace_third=(g(0, 0)+g(1, 1)+g(2, 2))/3.;
    Coordinate strain_axis{};
    double norm2=0.;
    // Тензор деформаций симметричен. Внедиагональный элемент вычисляем
    // один раз и учитываем в обеих компонентах произведения S·axis.
    for (unsigned i=0;i<3;++i) {
        const double diagonal=g(i, i)-trace_third;
        strain_axis[i]+=diagonal*axis[i];
        norm2+=diagonal*diagonal;
        for (unsigned j=i+1;j<3;++j) {
            const double shear=.5*(g(i, j)+g(j, i));
            strain_axis[i]+=shear*axis[j];
            strain_axis[j]+=shear*axis[i];
            norm2+=2.*shear*shear;
        }
    }
    const Coordinate tilt=cross3(strain_axis,axis);
    return norm2>0. ? ::sqrt(2.*dot3(tilt,tilt)/norm2) : 0.;
}
AERO_HOST_DEVICE inline double kh_factor(double mean_vtm, double lower) {
    const double ramp=::fmax(0.,::fmin(1.,(mean_vtm-.15)/.15));
    return lower+(1.-lower)*ramp;
}
// В одной ячейке это квадрат масштаба с VTM или вязкость Sigma с тем же VTM.
struct LesSample { double value, tilt; };
AERO_HOST_DEVICE inline double normalize_gradient(VelocityGradient& g) {
    // В определителе G^T G появляются шестые степени производных. Для обычных
    // величин они представимы в double. Крайне большие или малые градиенты
    // нормируем, а после вычисления возвращаем частоте её физический масштаб.
    double largest=0.;
    for (unsigned i=0;i<3;++i) for (unsigned j=0;j<3;++j) {
        const double v=::fabs(g(i, j));
        if (v>largest) largest=v;
    }
    if (largest==0.) return 0.;
    double rate_scale=1.;
    if (largest<1.e-16 || largest>1.e16) {
        rate_scale=largest;
        if (largest>1.e-300) {
            const double inverse=1./largest;
            for (unsigned i=0;i<3;++i) for (unsigned j=0;j<3;++j) g(i, j)*=inverse;
        } else {
            for (unsigned i=0;i<3;++i) for (unsigned j=0;j<3;++j) g(i, j)/=largest;
        }
    }
    return rate_scale;
}

// Ориентацию фильтра и VTM считаем из того же среднего градиента элемента в Sigma и DDES.
template<class Geometry>
AERO_HOST_DEVICE inline LesSample filter_scale(const VelocityGradient& g, const Geometry& cell,
                                                unsigned count, const LesSettings& settings, double rotation_floor) {
    if (settings.scale==Scale::Min || settings.scale==Scale::Max || settings.scale==Scale::Vol)
        return {cell.length_squared(settings.scale,count,Coordinate{}),0.};
    Coordinate axis{{g(2, 1)-g(1, 2),g(0, 2)-g(2, 0),g(1, 0)-g(0, 1)}};
    const double square=dot3(axis,axis);
    const double rotation=square>1.e-290 ? ::sqrt(square) : coordinate_length(axis);
    // У почти безвихревого поля направление может задаваться одной ошибкой округления
    // при сложении вкладов вершин. Как и для точно нулевого ротора, берём объёмный масштаб.
    if (rotation<=rotation_floor) return {cell.vol_length*cell.vol_length,0.};
    // Одна нормировка для трёх компонент. Если обратная длина переполняется,
    // деление каждой компоненты на длину сохраняет направление слабого вихря.
    if (rotation >= DBL_MIN) {
        const double inverse = reciprocal(rotation);
        for (unsigned i=0; i<3; ++i) axis[i] *= inverse;
    } else {
        for (unsigned i=0; i<3; ++i) axis[i] /= rotation;
    }
    const double delta2=cell.length_squared(settings.scale,count,axis);
    const double tilt=settings.scale==Scale::Sla && settings.switch_on_vtm ? vortex_tilting(g,axis) : 0.;
    return {delta2,tilt};
}

template<class Geometry>
AERO_HOST_DEVICE inline LesSample evaluate_sigma(VelocityGradient& g, const Geometry& cell,
                                                 unsigned count, const LesSettings& settings, double rotation_floor=0.) {
    const double magnitude=normalize_gradient(g);
    if (magnitude==0.) return {};
    const auto filter=filter_scale(g,cell,count,settings,rotation_floor/magnitude);
    return {settings.c_sigma*settings.c_sigma*filter.value*(magnitude*sigma_rate(g)),filter.tilt};
}
}
}
