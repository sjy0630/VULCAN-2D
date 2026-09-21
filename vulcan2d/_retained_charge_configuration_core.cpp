// Reuse the archived, tested frozen-state load solver without changing it.
#include "_activated_configuration_core.cpp"

static const int RC_FROM[8]={0,1,2,3,0,2,1,3};
static const int RC_TO[8]={1,0,3,2,2,0,3,1};

static void rc_identity(double* a) {
    for(int i=0;i<16;i++)a[i]=0.;for(int i=0;i<4;i++)a[5*i]=1.;
}
static void rc_multiply(const double* a,const double* b,double* c) {
    double temp[16];
    for(int i=0;i<4;i++)for(int j=0;j<4;j++) {
        temp[4*i+j]=a[4*i]*b[j]+a[4*i+1]*b[4+j]+a[4*i+2]*b[8+j]+a[4*i+3]*b[12+j];
    }
    std::copy(temp,temp+16,c);
}
static void rc_normalize(double* a) {
    for(int j=0;j<4;j++) {
        double sum=a[j]+a[4+j]+a[8+j]+a[12+j];
        for(int i=0;i<4;i++)a[4*i+j]/=sum;
    }
}

extern "C" int rc_propagator(const double* logs,double duration,double* out) {
    if(duration==0.){rc_identity(out);return 0;}
    double largest=logs[0];for(int i=1;i<8;i++)largest=std::max(largest,logs[i]);
    double rates[8],totals[4]={0.,0.,0.,0.};
    for(int i=0;i<8;i++){rates[i]=std::exp(logs[i]-largest);totals[RC_FROM[i]]+=rates[i];}
    double norm=*std::max_element(totals,totals+4);
    double logtime=largest+std::log(norm)+std::log(duration);
    if(!std::isfinite(logtime))return 1;
    const double ln2=std::log(2.);
    int squarings=std::max(0,int(std::ceil((logtime-std::log(.5))/ln2)));
    if(squarings>2048)return 2;
    double lambda=std::exp(logtime-squarings*ln2);
    double transition[16]={0.};
    for(int i=0;i<4;i++)transition[5*i]=std::max(0.,1.-totals[i]/norm);
    for(int i=0;i<8;i++)transition[4*RC_TO[i]+RC_FROM[i]]+=rates[i]/norm;
    rc_normalize(transition);
    double power[16];rc_identity(power);
    double weight=std::exp(-lambda);for(int i=0;i<16;i++)out[i]=weight*power[i];
    // Positive uniformization at lambda<=0.5, followed by stochastic squaring.
    // Never subtract nearly equal equilibrium and transient probabilities.
    for(int order=1;order<40;order++) {
        rc_multiply(transition,power,power);weight*=lambda/order;
        for(int i=0;i<16;i++)out[i]+=weight*power[i];
        if(weight<1e-19)break;
    }
    rc_normalize(out);
    for(int j=0;j<squarings;j++){rc_multiply(out,out,out);rc_normalize(out);}
    for(int i=0;i<16;i++)if(!std::isfinite(out[i])||out[i]<0.)return 3;
    return 0;
}

struct RCAdvance {
    const Engine& engine; int k,nv; const std::vector<double>& tables;
    std::vector<double> apply(const std::vector<double>& state,double voltage,int duration,bool half) const {
        int i;double fraction;engine.coordinate(voltage,i,fraction);
        std::vector<double> out(4*k);
        for(int h=0;h<k;h++) {
            size_t left=(((size_t)duration*nv+i)*k+h)*32+(half?0:16);
            size_t right=left+k*32;
            for(int row=0;row<4;row++) {
                double value=0.;
                for(int col=0;col<4;col++) {
                    int index=4*row+col;
                    double probability=(1.-fraction)*tables[left+index]+fraction*tables[right+index];
                    value+=probability*state[col*k+h];
                }
                out[row*k+h]=value;
            }
        }
        return out;
    }
};

extern "C" int rc_run(int n,int k,int nv,const double* applied,const double* grid,
    const double* j0,const double* j1,const double* rates,const double* weights,
    int nd,const double* durations,const int* duration_ids,const int* nsub,
    int np,const double* pv,const double* pi,int nn,const double* nvds,const double* ni,
    const double* initial,double* current,double* vh,double* vd,double* states,
    double* errors,double* roots,int diagnostics) {
    Engine engine{2*k,nv,np,nn,grid,j0,j1,nullptr,weights,pv,pi,nvds,ni,grid[1]-grid[0]};
    std::vector<double> tables((size_t)nd*nv*k*32);
    for(int d=0;d<nd;d++)for(int i=0;i<nv;i++)for(int h=0;h<k;h++) {
        double* half=&tables[(((size_t)d*nv+i)*k+h)*32];
        int code=rc_propagator(rates+((size_t)i*k+h)*8,durations[d]/2,half);
        if(code)return 10+code;
        rc_multiply(half,half,half+16);rc_normalize(half+16);
    }
    RCAdvance advance{engine,k,nv,tables};std::vector<double> state(initial,initial+4*k);
    double previous=0.;
    for(int j=0;j<n;j++) {
        if(j>0&&nsub[j]>0) {
            double delta=applied[j]-applied[j-1];int sub=nsub[j],d=duration_ids[j];
            for(int z=0;z<sub;z++) {
                double middle=applied[j-1]+delta*(z+.5)/sub;
                auto first=engine.load(middle,state,previous,false);
                if(!std::isfinite(first.current))return 1;
                auto half=advance.apply(state,first.vh,d,true);
                auto midpoint=engine.load(middle,half,first.vh,false);
                if(!std::isfinite(midpoint.current))return 2;
                state=advance.apply(state,midpoint.vh,d,false);
                auto end=engine.load(applied[j-1]+delta*(z+1.)/sub,state,midpoint.vh,false);
                if(!std::isfinite(end.current))return 3;
                previous=end.vh;
            }
        }
        auto out=engine.load(applied[j],state,previous,diagnostics!=0);
        if(!std::isfinite(out.current))return 4;
        current[j]=out.current;vh[j]=out.vh;vd[j]=out.vd;errors[j]=out.error;roots[j]=out.roots;
        std::copy(state.begin(),state.end(),states+j*4*k);previous=out.vh;
    }
    return 0;
}
