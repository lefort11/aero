#ifndef AERO_EXPLICIT_STAGE_SEQUENCE_H
#define AERO_EXPLICIT_STAGE_SEQUENCE_H
#include "explicit.h"
#include "explicit_stage.h"
#include <string>

namespace aero { namespace explicit_detail {
ExplicitScheme scheme_from_name(const std::string& name);
Stages stages_for(ExplicitScheme scheme);

// A CFL calculation can supply the first residual before dt is known.
// Finish the halo exchange before the next assembly or the accepted-step output.
template<class Assemble,class Update,class Exchange>
void advance_stages(Stages stages,double time,double dt,bool first_prepared,
                    const Assemble& assemble,const Update& update_stage,const Exchange& exchange) {
    for (unsigned i=0;i<stages.count;++i) {
        const Stage stage=stages.data[i];
        const bool first=i==0,last=i+1==stages.count;
        if (!(first && first_prepared)) assemble(time+stage.time_fraction*dt);
        update_stage(stage,first,last);
        exchange();
    }
}
}}
#endif
