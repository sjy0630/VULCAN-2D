// Reuse archived load root logic; this module introduces no state dynamics.
#include "_activated_configuration_core.cpp"

extern "C" int fr_load(int nc,int n,int nv,const double* fractions,
    const double* applied,const double* grid,const double* currents,
    int np,const double* pv,const double* pi,int nn,const double* mv,const double* mi,
    double* current,double* internal,double* transistor,double* error) {
    double weights[4]={1.,1.,1.,1.};
    Engine engine{4,nv,np,nn,grid,currents,currents,nullptr,weights,pv,pi,mv,mi,grid[1]-grid[0]};
    for(int c=0;c<nc;c++) {
        std::vector<double> population(8,0.);
        for(int j=0;j<4;j++)population[j]=fractions[c*4+j];
        double previous=0.;
        for(int j=0;j<n;j++) {
            auto result=engine.load(applied[j],population,previous,false);
            if(!std::isfinite(result.current))return 1;
            size_t i=(size_t)c*n+j;
            current[i]=result.current;internal[i]=result.vh;transistor[i]=result.vd;error[i]=result.error;
            previous=result.vh;
        }
    }
    return 0;
}
