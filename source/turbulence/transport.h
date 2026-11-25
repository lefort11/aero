#pragma once
#include "../timestep/node_step.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#include "../io/diagnostic_fields.h"
#include "diffusion_geometry.h"
#include "face_geometry.h"

namespace aero {
namespace turbulence {
// Все индексы и указатели относятся к одной подобласти и одному исполнителю.
struct Transport {
    // Узловая оценка масштаба турбулентности.
    NodalFilter filter;
    GasFaces gas_faces; // При компактном размещении SA не хранит вторую таблицу тех же граней.
    View sa; // Только массивы, нужные уравнению SA на выбранном исполнителе.
    // Газовые данные для конвекции и источников SA.
    const viscosity::Gas *gas_nodes;
    // Связность и коэффициенты элементов.
    viscosity::CellTable cells;
    // Число элементов и внутренних рёбер.
    MeshIndex cell_count, edges;
    // Размерность задачи переноса.
    unsigned dim;
    // Массовые потоки газа через площадки.
    const EulerFluxResult *flux;
    const double* convection_correction; // Один ориентированный поток на внутреннее ребро, nullptr при первом порядке.
    const ProfileState* profiles; // Входное nu_tilde на физических площадках.
    // Узловые контрольные объёмы.
    const double *volume;
    // Соответствие собственной строки узлу.
    const MeshIndex *owned;
    // Число обрабатываемых строк переноса.
    MeshIndex count;
    viscosity::GradientStencil shared_geometry; // Таблица коэффициентов AES на том же исполнителе.
    double heat_to_diffusion; // Перевод теплового коэффициента AES в коэффициент диффузии SA.

    // Переводит номер обрабатываемой строки в номер собственного узла.
    AERO_HOST_DEVICE MeshIndex node(MeshIndex k) const { return owned ? owned[k] : k; }
    // В уравнении переноса разность с соседним значением нужна только при втекании.
    AERO_HOST_DEVICE double incoming_speed(residual_detail::OrientedFace side, double inverse_density) const {
        const double inward = -side.outward_flux(flux[side.face()].flux[0]) * inverse_density;
        return inward > 0. ? inward : 0.;
    }
    // На CPU одна разность nu_tilde и один коэффициент обслуживают обе вершины
    // пары с противоположными знаками. Элементы независимы: каждый сохраняет
    // только свои восемь вкладов. Владельцы узлов затем суммируют их без атомиков.
    template<bool refresh=false>
    AERO_HOST_DEVICE void diffuse_element(MeshIndex e) const {
        const auto& cell=cells[e];
        const auto& stencil=sa.diffusion;
        const double* weights=stencil.pair_weights+stencil.shape[e];
        double working[8], sum[8]{};
        double working_sum=0.;
        for (unsigned i=0;i<cell.count;++i) {
            working[i]=sa.w[cell.node[i]];
            if (refresh) working_sum+=working[i];
        }
        if (refresh) set_diffusivity(e,working_sum);
        for (unsigned i=0;i<cell.count;++i) for (unsigned j=i+1;j<cell.count;++j) {
            const double flow=(*weights++)*(working[j]-working[i]);
            sum[i]+=flow;
            sum[j]-=flow;
        }
        const double mean=stencil.element_mean[e];
        for (unsigned i=0;i<cell.count;++i)
            stencil.vertex_contributions[8*e+i]={mean*sum[i],::fabs(mean)*weights[i]};
    }
    // Значение вычисляется один раз на элемент и используется во всех прилегающих строках.
    AERO_HOST_DEVICE void set_diffusivity(MeshIndex e,double working_sum) const {
        sa.diffusion.element_mean[e]=sa.diffusion.molecular_mean[e]+working_sum*cells[e].inverse_count;
    }
    // Обновляет коэффициенты диффузии по текущей рабочей вязкости.
    template<bool host=false>
    AERO_HOST_DEVICE void refresh_diffusivity(MeshIndex e) const {
        if (host && sa.diffusion.vertex_contributions) { diffuse_element<true>(e); return; }
        const auto &cell = cells[e];
        double sum = 0.;
        for (unsigned j = 0; j < cell.count; ++j) {
            const auto n = cell.node[j];
            sum += sa.w[n];
        }
        set_diffusivity(e,sum);
    }
    // Для обычных подобластей номера и смещения умещаются в 32 бита.
    // Беззнаковое сложение восстанавливает номер соседа в обоих форматах.
    template<class Index>
    AERO_HOST_DEVICE void add_pairs(const DiffusionEntry<Index>* terms,RowRange row,unsigned stride,
                                    MeshIndex first,MeshIndex center,double w,double& diffusion,double& diagonal) const {
        const Index e0=static_cast<Index>(first), n0=static_cast<Index>(center);
        for (auto pos=row.first;pos<row.end;pos+=stride) {
            const auto term=terms[pos];
            const double coupling=sa.diffusion.element_mean[e0+term.element]*term.geometry;
            diffusion+=coupling*(sa.w[n0+term.neighbour]-w);
            diagonal+=::fabs(coupling);
        }
    }
    // Добавляет вклад упакованной плотной диффузионной строки.
    template<unsigned neighbours>
    AERO_HOST_DEVICE void add_dense(const DiffusionCell* blocks,RowRange row,double w,
                                    double& diffusion,double& diagonal) const {
        for (auto pos=row.first;pos<row.end;++pos) {
            const auto& block=blocks[pos];
            const double mean=sa.diffusion.element_mean[block.element];
            double jump=0.;
            for (unsigned j=0;j<neighbours;++j)
                jump+=block.geometry[j]*(sa.w[block.node[j]]-w);
            diffusion+=mean*jump;
            diagonal+=::fabs(mean)*block.diagonal;
        }
    }
    // У треугольника и четырёхугольника не более трёх соседей в элементе.
    // Таблица хранит семь мест для смешанных сеток. Здесь длина фактического
    // цикла известна при компиляции, поэтому нулевые заполнители не загружаются.
    AERO_HOST_DEVICE void element_diffusion(const DiffusionCell* blocks,RowRange row,double w,
                                            double& diffusion,double& diagonal) const {
        if (sa.diffusion.short_blocks) add_dense<3>(blocks,row,w,diffusion,diagonal);
        else add_dense<7>(blocks,row,w,diffusion,diagonal);
    }
    // Добавляет диффузионные вклады объёмных элементов AES.
    template<bool host>
    AERO_HOST_DEVICE void add_volume_diffusion(MeshIndex i,double w,double& diffusion,double& diagonal) const {
        // Каждая ненулевая пара даёт приращение от разности рабочих вязкостей.
        // Коэффициенты могут иметь разный знак на косой сетке, поэтому в оценке
        // шага берём модуль каждого вклада до суммирования по элементам.
        const auto &stencil = sa.diffusion;
        if (!host) {
            diffuse_from_aes(i,w,diffusion,diagonal);
            return;
        }
        if (stencil.vertex_contributions) {
            for (auto k=filter.offsets[i];k<filter.offsets[i+1];++k) {
                const auto entry=filter.incidence[k];
                const auto contribution=stencil.vertex_contributions[entry];
                diffusion+=contribution.rate;
                diagonal+=contribution.diagonal;
            }
            return;
        }
        const auto row=stencil.rows[i];
        if (stencil.cell_rows) { element_diffusion(stencil.cell_rows,row,w,diffusion,diagonal); return; }
        const auto base_element = stencil.anchor ? stencil.anchor[i] : 0;
        const auto base_node = stencil.anchor ? i : 0;
        const unsigned stride=1;
        if (stencil.triples) {
            // Три разности имеют общий множитель. Сумма модулей геометрических
            // коэффициентов для диагонали уже известна на неподвижной сетке.
            // Сложение смещений без знака восстанавливает номер по модулю 2^32.
            const auto first=static_cast<std::uint32_t>(base_element);
            const auto center=static_cast<std::uint32_t>(base_node);
            for (auto pos=row.first;pos<row.end;pos+=stride) {
                const auto& group=stencil.triples[pos];
                const double mean=stencil.element_mean[first+group.element];
                double jump=0.;
                for (unsigned q=0;q<3;++q)
                    jump+=group.geometry[q]*(sa.w[center+group.neighbour[q]]-w);
                diffusion+=mean*jump;
                diagonal+=::fabs(mean)*group.diagonal;
            }
        } else if (stencil.local_terms) {
            add_pairs(stencil.local_terms,row,stride,base_element,base_node,w,diffusion,diagonal);
        } else {
            add_pairs(stencil.terms,row,stride,base_element,base_node,w,diffusion,diagonal);
        }
    }
    // Обход соответствует порядку заполнения плиток: сначала элементы узла,
    // внутри элемента — вершины по исходному номеру. Множитель вязкости
    // применяем к сумме по элементу один раз, как и при чтении готовой плитки.
    AERO_HOST_DEVICE void diffuse_from_aes(MeshIndex node,double w,double& rate,double& diagonal) const {
        for (auto p=filter.offsets[node]; p<filter.offsets[node+1]; ++p) {
            const auto entry=filter.incidence[p], e=entry/8;
            const unsigned local=static_cast<unsigned>(entry%8);
            const auto& cell=cells[e];
            double jump=0., absolute_sum=0.;
            for (unsigned j=0; j<cell.count; ++j) {
                const double weight=diffusion_weight(shared_geometry,cell,local,j,heat_to_diffusion);
                if (weight==0.) continue;
                jump+=weight*(sa.w[cell.node[j]]-w);
                absolute_sum+=::fabs(weight);
            }
            const double mean=sa.diffusion.element_mean[e];
            rate+=mean*jump;
            diagonal+=::fabs(mean)*absolute_sum;
        }
    }
    // Схема первого порядка по потоку массы сохраняет постоянное nu_tilde и при
    // переменной плотности. На втекании используем набегающий поток, на вытекании — значение узла.
    template <unsigned d, bool reuse_gas, bool host, bool axial,bool shared=false>
    AERO_HOST_DEVICE void assemble_sa_row(MeshIndex k, bool save_rotation) const {
        const auto i = node(k);
        // Первый обход на новом газе собирает длину DDES в том же узле, где
        // она нужна источнику. В повторных итерациях SA меняется только nu_tilde.
        if (host && !reuse_gas && filter.points && sa.inverse_distance[i]>=0.) filter.gather<true>(i);
        if (sa.zero_working_viscosity(i)) {
            sa.rate[i] = sa.diagonal[i] = 0.;
            return;
        }
        const auto &gas = gas_nodes[i];
        const double w = sa.w[i], nu = sa.nodal_material[i].nu;
        double advection = 0., diffusion = 0., diagonal = 0., dw[3]{};
        diagnostic::Curl curl{};
        const double inverse_density = reciprocal(gas.density);
        const auto faces = shared ? gas_faces.row(i) : sa.face_rows[i];
        for (auto pos = faces.first; pos < faces.end; pos += (host ? 1 : sa.face_stride)) {
            const AxisFace compact=axial ? sa.axis_faces[pos] : AxisFace{};
            const FaceTerm general = shared ? gas_faces.read(i,pos,edges,d) :
                                     !axial && !host && sa.face_columns.area ? sa.face_columns.read<d>(pos) :
                                     axial ? FaceTerm{} : sa.face_terms[pos];
            const FaceTerm* face=&general;
            const residual_detail::OrientedFace orientation{axial ? compact.side_axis/4 : face->encoded};
            const bool edge = orientation.face() < edges;
            const auto j = axial ? compact.neighbour : face->neighbour;
            // В серии скалярных итераций поток газа на ребре один и тот же.
            const double incoming = reuse_gas && sa.incoming ? sa.incoming[pos] : incoming_speed(orientation, inverse_density);
            if (!reuse_gas && save_rotation && sa.incoming)
                sa.incoming[pos] = incoming;
            double jump = 0.;
            if (edge) jump = sa.w[j] - w;
            else if (incoming > 0.) {
                const auto index = orientation.face()-edges;
                double inlet = sa.ambient;
                if (profiles && profiles[index].present()) {
                    const auto& sample = profiles[index];
                    if (!sample.columns || (sample.columns & NuTilde)) inlet = sample.nutilde;
                    else if (sample.fallback == ProfileFallback::Interior) continue; // inlet=w: и разность, и её производная равны нулю.
                    else if (sample.fallback == ProfileFallback::Initial && sa.inverse_distance[i] < 0.) inlet = 0.;
                }
                jump = inlet-w;
            }
            advection += incoming * jump;
            diagonal += incoming;
            if (convection_correction && edge) {
                // Владелец ребра вычислил добавочный поток один раз. После смены
                // нормали два узла получают противоположные вклады в перенос rho*nu_tilde.
                const double outward = orientation.outward_flux(convection_correction[orientation.face()]) * inverse_density;
                advection -= outward;
                // При явном шаге учитываем дополнительный отбор из узла в ограничении CFL.
                // На входящей грани неотрицательность уже обеспечивает неотрицательное значение на грани.
                if (!reuse_gas && !save_rotation && incoming == 0. && outward > 0. && w > 0.)
                    diagonal += outward / w;
            }
            double jump_u[3]{};
            if (!reuse_gas && edge)
                for (unsigned n=0;n<d;++n) jump_u[n]=gas_nodes[j].velocity[n]-gas.velocity[n];
            if (axial) {
                compact.add_derivatives<!reuse_gas>(jump,jump_u,dw,curl.value);
            } else {
                for (unsigned m=0;m<d;++m) dw[m]+=face->area[m]*jump;
                if (!reuse_gas && edge) curl.add(face->area,jump_u,1.);
            }
        }
        const double inv_volume = reciprocal(volume[i]);
        double grad2 = 0.;
        for (unsigned m = 0; m < d; ++m) {
            const double derivative = dw[m] * inv_volume;
            grad2 += derivative * derivative;
        }
        add_volume_diffusion<host>(i,w,diffusion,diagonal);
        // Пока Ньютон уточняет только рабочую вязкость, поле скорости неизменно.
        const double rotation = reuse_gas ? sa.rotation[i] : curl.magnitude(inv_volume);
        if (!reuse_gas && save_rotation)
            sa.rotation[i] = rotation;
        // В DDES меняется длина во всех членах источника SA. Скорость фиксирована
        // на внутренних итерациях BDF, но ν_t пересчитываем по текущему nu_tilde.
        double inverse=sa.inverse_distance[i];
        const double chi=w/nu;
        // В явной стадии fv1 уже использовалась в вязком потоке газа.
        // При уточнении BDF рабочая вязкость меняется и демпфирование считаем заново.
        const double damp=!reuse_gas && !save_rotation ? sa.nodal_material[i].damping :
                          (inverse!=0. || sa.ddes ? fv1(chi) : 0.);
        if (sa.ddes) {
            inverse=sa.ddes[i].inverse_length(inverse,nu+w*damp,kappa);
            if (inverse<0.) {
                sa.rate[i]=sa.diagonal[i]=0.;
                return;
            }
        }
        const double source=source_rate_from_chi(w,chi,damp,rotation,inverse);
        sa.rate[i] = (advection + diffusion) * inv_volume + cb2 / sigma * grad2 + source * w;
        sa.diagonal[i] = diagonal * inv_volume + ::fabs(source);
    }
    // Геометрия площадок едина для всей подобласти. У осевой площадки
    // нужна одна компонента площади, у произвольной — все компоненты.
    template<unsigned d,bool reuse_gas,bool host=false>
    AERO_HOST_DEVICE void residual_row(MeshIndex k,bool save_rotation) const {
        if (!host && !sa.face_rows) assemble_sa_row<d,reuse_gas,host,false,true>(k,save_rotation);
        else if (sa.axis_faces) assemble_sa_row<d,reuse_gas,host,true>(k,save_rotation);
        else assemble_sa_row<d,reuse_gas,host,false>(k,save_rotation);
    }
    // Вычисляет полную невязку переноса SA для выбранного узла.
    template <bool reuse_gas> AERO_HOST_DEVICE void residual(MeshIndex k, bool save_rotation) const {
        if (dim == 3)
            residual_row<3, reuse_gas>(k, save_rotation);
        else if (dim == 2)
            residual_row<2, reuse_gas>(k, save_rotation);
        else
            residual_row<1, reuse_gas>(k, save_rotation);
    }
};
enum class Operation {
    Residual,
    Explicit,
    SaveStart, RestoreStart,
    SaveBackup,
    RestoreBackup,
    CommitHistory,
    Relax,
    RestorePlain,
    LineRelax,
    PredictRate // Начальное приближение steady по пространственным скоростям на принятых слоях.
};
// При первом контроле BDF можно за тот же обход сохранить начало пробного шага.
enum class Norm { Rate, Bdf, SaveStartBdf };
// Поле и временные коэффициенты одной явной или неявной стадии переноса SA.
struct Step {
    // Пространственный оператор рабочего поля.
    Transport transport;
    // Стадия явного интегратора.
    explicit_detail::Stage stage;
    // Шаг и коэффициенты временных слоёв неявной формулы.
    double dt, leading, history;
    // Узловые псевдошаги стационарного режима.
    NodeSteps local_steps;
    // Требуется сохранить поле начала стадии.
    bool save_stage_start;
    bool extrapolate; // Разрешает ускорение двух последовательных скалярных поправок BDF.
    bool keep_gas; // Давление, плотность и скорость фиксированы между итерациями скалярного уравнения.
    double prediction_ratio = 0.; // Отношение текущего минимального псевдошага к предыдущему.
#ifdef AERO_HAS_MULTIGRID
    double polynomial_current, polynomial_previous; // Нулевой первый коэффициент означает обычный сглаживатель.
#endif
    // Выбирает обход на хосте до запуска CPU-команды или GPU-ядра.
    template<class Lines, class Residual, class Nodes>
    void dispatch(Operation op, const Lines& lines, const Residual& residual, const Nodes& nodes) const {
        if (op==Operation::LineRelax && transport.sa.lines.count) lines(transport.sa.lines.count);
        else if (op==Operation::Residual) residual();
        else nodes(transport.count);
    }
    // CPU может совместить обновление строк с нормой, GPU — выполнить отдельные ядра.
    template<class Diffusivity, class UpdatedNorm, class CurrentNorm>
    double evaluate_norm(Norm mode, bool refresh, const Diffusivity& diffusivity,
                         const UpdatedNorm& updated, const CurrentNorm& current) const {
        if (!refresh) return current(mode);
        diffusivity();
        return updated();
    }
    // Погрешность дискретного уравнения на новом слое BDF в единицах рабочей вязкости.
    AERO_HOST_DEVICE double defect(MeshIndex i) const {
        const auto &s = transport.sa;
        const double previous = history ? history * (s.old[i] - s.previous[i]) : 0.;
        return leading * (s.w[i] - s.old[i]) - previous - local_steps.at(i, dt) * s.rate[i];
    }
    // Применяет неявную поправку к рабочей вязкости с учётом диагонали и границ.
    AERO_HOST_DEVICE double implicit_update(MeshIndex i) const {
        auto& s=transport.sa;
        const double delta=-defect(i)/(leading+local_steps.at(i,dt)*s.diagonal[i]);
        const double plain=s.w[i]+delta;
        if (!extrapolate) {
            s.last_correction[i]=delta;
            return plain;
        }
        s.plain_update[i]=s.zero_working_viscosity(i) || plain<0. ? 0. : plain;
#ifdef AERO_HAS_MULTIGRID
        if (polynomial_current > 0.) {
            // Все узлы используют одни коэффициенты. Так ускорение сохраняет
            // пространственную форму ошибки, в том числе когда поправки меняют знак.
            double change = polynomial_current*delta;
            if (polynomial_previous != 0.) change += polynomial_previous*s.last_correction[i];
            const double next = s.w[i]+change;
            const double projected = s.zero_working_viscosity(i) || next<0. ? 0. : next;
            s.last_correction[i] = projected-s.w[i];
            return projected;
        }
#endif
        return extrapolated_viscosity(plain,delta,s.last_correction[i]);
    }
    // На вытянутой сетке учитываем две сильные связи в поправке совместно.
    // Правая часть по-прежнему берётся из полной невязки SA со всеми соседями.
    AERO_HOST_DEVICE void relax_line(MeshIndex segment) const {
        auto& sa=transport.sa;
        const auto rows=sa.lines.segments[segment];
        const unsigned count=static_cast<unsigned>(rows.end-rows.first);
        double upper[SaLines::length], correction[SaLines::length];
        for (unsigned k=0;k<count;++k) {
            const auto vertex=sa.lines.vertices[rows.first+k];
            const auto i=vertex.node;
            const double h=local_steps.at(i,dt);
            const double diagonal=leading+h*sa.diagonal[i];
            const double residual=sa.zero_working_viscosity(i) ? 0. : -defect(i);
            const double plain=sa.w[i]+residual/diagonal;
            sa.plain_update[i]=sa.zero_working_viscosity(i) || plain<0. ? 0. : plain;
            double scale=h*(sa.nodal_material[i].nu+sa.w[i])/transport.volume[i];
            const double sum=vertex.left+vertex.right;
            // Неточная производная ограничена фактической диагональю переноса.
            // Временной член leading сохраняет строгое диагональное преобладание.
            if (sum>0.) scale=::fmin(scale,h*sa.diagonal[i]/sum);
            const double left=scale*vertex.left;
            const double pivot=diagonal-(k ? left*upper[k-1] : 0.);
            upper[k]=scale*vertex.right/pivot;
            correction[k]=(residual+(k ? left*correction[k-1] : 0.))/pivot;
        }
        for (unsigned end=count;end>0;--end) {
            const unsigned k=end-1;
            if (end<count) correction[k]+=upper[k]*correction[k+1];
            const auto i=sa.lines.vertices[rows.first+k].node;
            const double next=sa.w[i]+correction[k];
            sa.w[i]=sa.zero_working_viscosity(i) || next<0. ? 0. : next;
        }
    }
    // Выполняет выбранное обновление одной компоненты поля SA.
    AERO_HOST_DEVICE void apply(Operation op, MeshIndex k) const {
        const auto i = transport.node(k);
        const auto &s = transport.sa;
        switch (op) {
        case Operation::Residual:
            transport.residual<false>(k, keep_gas);
            break;
        case Operation::SaveStart:
            s.old[i] = s.w[i];
            break;
        case Operation::PredictRate: {
            // В steady буфер предыдущего слоя BDF2 свободен. Начальная скорость
            // берётся до изменения газа, новый баланс SA проверяется после пробы.
            const double rate = s.rate[i];
            const double predicted = prediction_ratio ? rate + prediction_ratio * (rate-s.previous[i]) : rate;
            const double next = s.old[i] + local_steps.at(i,dt) * predicted;
            s.previous[i] = rate;
            s.w[i] = s.zero_working_viscosity(i) ? 0. : (::fabs(next)<=max_finite_double ? ::fmax(0.,next) : s.old[i]);
            break;
        }
        case Operation::RestoreStart:
            s.w[i] = s.old[i];
            break;
        case Operation::SaveBackup:
            s.backup[i] = s.w[i];
            break;
        case Operation::RestoreBackup:
            s.w[i] = s.backup[i];
            break;
        case Operation::CommitHistory:
            s.previous[i] = s.old[i];
            break;
        case Operation::LineRelax: {
            // На ранге без сильных связей та же проба совпадает с обычной поправкой.
            const double plain=s.w[i]-defect(i)/(leading+local_steps.at(i,dt)*s.diagonal[i]);
            s.plain_update[i]=s.w[i]=s.zero_working_viscosity(i) || plain<0. ? 0. : plain;
            break;
        }
        case Operation::RestorePlain:
            s.w[i]=s.plain_update[i];
            break;
        case Operation::Explicit:
            s.advance_explicit(i,stage,dt,save_stage_start);
            break;
        case Operation::Relax: {
            const double next=implicit_update(i);
            s.w[i] = s.zero_working_viscosity(i) ? 0. : (next < 0. ? 0. : next);
            break;
        }
        }
    }
    // После изменения nu_tilde считаем строку заново и сразу используем
    // полученную правую часть для нормы BDF. Отдельный проход по полю не нужен.
    template<unsigned dim,bool host>
    AERO_HOST_DEVICE double updated_defect(MeshIndex k) const {
        transport.residual_row<dim,true,host>(k,false);
        return magnitude(Norm::Bdf,k);
    }
    // Возвращает модуль величины SA, выбранной для контроля сходимости.
    AERO_HOST_DEVICE double magnitude(Norm norm, MeshIndex k) const {
        const auto i = transport.node(k);
        const auto &s = transport.sa;
        if (norm == Norm::SaveStartBdf) {
            s.old[i] = s.w[i];
            norm = Norm::Bdf;
        }
        if (norm==Norm::Bdf && s.zero_working_viscosity(i)) return ::fabs(s.w[i]);
        double value = norm == Norm::Bdf ? defect(i) : s.rate[i];
        // При нулевом значении рабочей вязкости допускается только неотрицательное обновление.
        if (norm == Norm::Bdf && s.w[i] == 0. && value > 0.)
            value = 0.;
        return ::fabs(value) <= max_finite_double ? ::fabs(value) : HUGE_VAL;
    }
};
}
}
