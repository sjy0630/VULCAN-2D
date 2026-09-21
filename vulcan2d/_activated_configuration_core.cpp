#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

struct LoadResult {double vh, vd, current, error; int roots;};
struct Engine {
    int k,nv,np,nn; const double *v,*j0,*j1,*rates,*w,*pv,*pi,*nvds,*ni;
    double step;
    double interp(double x,const double* xs,const double* ys,int n) const {
        if(x<=xs[0]) return ys[0]; if(x>=xs[n-1]) return ys[n-1];
        int hi=std::upper_bound(xs,xs+n,x)-xs,lo=hi-1;
        return ys[lo]+(ys[hi]-ys[lo])*(x-xs[lo])/(xs[hi]-xs[lo]);
    }
    void coordinate(double x,int& i,double& f) const {
        double pos=(x-v[0])/step; i=std::max(0,std::min(nv-2,int(std::floor(pos))));
        f=std::max(0.,std::min(1.,pos-i));
    }
    double device(double signed_v,const std::vector<double>& q) const {
        int i; double f; coordinate(signed_v,i,f); double total=0.;
        for(int h=0;h<k;h++) {
            double a=j0[i*k+h]+f*(j0[(i+1)*k+h]-j0[i*k+h]);
            double b=j1[i*k+h]+f*(j1[(i+1)*k+h]-j1[i*k+h]);
            total+=w[h]*(q[h]*a+q[k+h]*b);
        }
        return total;
    }
    double next_node(double x,double va,const double* tv,int nt,bool right) const {
        double node;
        if(right) {
            node=std::min(va,step*(std::floor(x/step+1e-10)+1));
            double d=va-x;
            int j=int(std::lower_bound(tv,tv+nt,d-1e-13)-tv)-1;
            if(j>=0) node=std::min(node,va-tv[j]);
            if(node<=x+1e-14) node=va;
        } else {
            node=std::max(0.,step*(std::ceil(x/step-1e-10)-1));
            double d=va-x;
            int j=int(std::upper_bound(tv,tv+nt,d+1e-13)-tv);
            if(j<nt) node=std::max(node,va-tv[j]);
            if(node>=x-1e-14) node=0.;
        }
        return node;
    }
    LoadResult load(double applied,const std::vector<double>& q,double previous,bool count_roots) const {
        if(applied==0.) return {0.,0.,0.,0.,1};
        double sign=applied>0?1.:-1.,va=std::abs(applied);
        const double *tv=sign>0?pv:nvds,*ti=sign>0?pi:ni; int nt=sign>0?np:nn;
        auto diff=[&](double h){return device(sign*h,q)-interp(va-h,tv,ti,nt);};
        double h=std::max(0.,std::min(va,sign*previous)); double f=diff(h),root=h,vd=va-h;
        bool right=f<=0.; bool found=f==0.;
        for(int it=0;!found&&it<nv+nt+10;it++) {
            double next=next_node(h,va,tv,nt,right),fn=diff(next);
            if((right&&fn>=0.)||(!right&&fn<=0.)) {
                // Solve the affine interval in transistor voltage, avoiding
                // subtraction of an almost-full memristor voltage for tiny I.
                double d0=va-h,d1=va-next;
                vd=(d1*f-d0*fn)/(f-fn);
                root=va-vd; found=true;
            } else {h=next;f=fn;}
        }
        if(!found) return {NAN,NAN,NAN,NAN,0};
        double ih=device(sign*root,q),it=interp(vd,tv,ti,nt);
        int nr=-1;
        if(count_roots) {
            nr=0;double h0=0.,f0=diff(0.);
            for(int j=0;j<nv+nt+10&&h0<va;j++) {
                double h1=next_node(h0,va,tv,nt,true),f1=diff(h1);
                if(f0*f1<0.||f1==0.)nr++;
                if(h1<=h0)break;h0=h1;f0=f1;
            }
        }
        return {sign*root,vd,ih,std::abs(ih-it)/std::max(1e-300,std::max(ih,it)),nr};
    }
    std::vector<double> advance(const std::vector<double>& q,double vh,double ds) const {
        if(ds==0.) return q;
        int i;double f;coordinate(vh,i,f);std::vector<double> out(2*k);
        for(int h=0;h<k;h++) {
            double l01=rates[(2*i)*k+h]+f*(rates[(2*(i+1))*k+h]-rates[(2*i)*k+h]);
            double l10=rates[(2*i+1)*k+h]+f*(rates[(2*(i+1)+1)*k+h]-rates[(2*i+1)*k+h]);
            double delta=l01-l10,z=std::exp(-std::abs(delta)),eq0,eq1;
            if(delta>=0.){eq1=1./(1.+z);eq0=z/(1.+z);}else{eq0=1./(1.+z);eq1=z/(1.+z);}
            double total=std::max(l01,l10)+std::log1p(std::exp(-std::abs(delta)));
            double rate_time=std::exp(std::min(std::log(745.),total+std::log(ds)));
            double survival=std::exp(-rate_time),transfer=-std::expm1(-rate_time);
            out[h]=q[h]*survival+eq0*transfer;out[k+h]=q[k+h]*survival+eq1*transfer;
        }
        return out;
    }
};

extern "C" int ac_run(int n,int k,int nv,const double* applied,const double* grid,
    const double* j0,const double* j1,const double* rates,const double* weights,
    int np,const double* pv,const double* pi,int nn,const double* nvds,const double* ni,
    double max_step,double initial,double* current,double* vh,double* vd,double* states,
    double* errors,double* roots,int diagnostics) {
    Engine e{k,nv,np,nn,grid,j0,j1,rates,weights,pv,pi,nvds,ni,grid[1]-grid[0]};
    std::vector<double> q(2*k);for(int h=0;h<k;h++){q[h]=1.-initial;q[k+h]=initial;}
    double previous=0.;
    for(int j=0;j<n;j++) {
        if(j>0) {
            double delta=applied[j]-applied[j-1];
            int sub=std::max(1,int(std::ceil(std::abs(delta)/max_step-1e-10)));
            for(int z=0;z<sub;z++) {
                double ds=std::abs(delta)/sub;
                double middle=applied[j-1]+delta*(z+.5)/sub;
                auto first=e.load(middle,q,previous,false);
                if(!std::isfinite(first.current))return 1;
                auto half=e.advance(q,first.vh,ds/2);
                auto midpoint=e.load(middle,half,first.vh,false);
                if(!std::isfinite(midpoint.current))return 2;
                q=e.advance(q,midpoint.vh,ds);
                auto end=e.load(applied[j-1]+delta*(z+1.)/sub,q,midpoint.vh,false);
                if(!std::isfinite(end.current))return 3;
                previous=end.vh;
            }
        }
        auto out=e.load(applied[j],q,previous,diagnostics!=0);
        if(!std::isfinite(out.current))return 4;
        current[j]=out.current;vh[j]=out.vh;vd[j]=out.vd;errors[j]=out.error;roots[j]=out.roots;
        std::copy(q.begin(),q.end(),states+j*2*k);previous=out.vh;
    }
    return 0;
}
