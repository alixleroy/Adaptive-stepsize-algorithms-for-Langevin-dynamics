

// anisotropic with two dimensions, with stepsize larger where potential steeper 

#include <cstring>
#include <stdio.h>
#include <random>
#include <cmath>
#include <string>
#include <list>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <boost/random/random_device.hpp> //boost function
#include <boost/random/normal_distribution.hpp> //include normal distribution
#include <boost/random/mersenne_twister.hpp>
#include <boost/multi_array.hpp>
#include <chrono>

// Xtensor vector
// #include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"



using namespace std::chrono;
 
using namespace std;

// # include <math.h>
// # include <complex.h>
// # include <stdlib.h>

// # include <stdio.h>
// # include <time.h>
// # include "normal.h"

using namespace std;
#define m               0.2          // minimum step scale factor
#define M               1.             // maximum step scale factor
#define numsam          1          // number of sample
#define T               100000         // total number of trajectories

#define dt              0.025
#define tau             .1            

#define numruns         int(T/dt)         // total number of trajectories
#define gamma           .5            // friction coefficient
#define printskip       10

#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/fewtraj_ani_mod/smalldt"
#define k1              .1
#define k2              50.
#define k3              50.
#define k4              .1


///////////////////////////////////////
/// Bobsled Potential around x=4     //
///////////////////////////////////////


double Upx(double x, double y){

    double x2,x3,x4,y2,p1,p2,p1x,p1y,p2x,p2y,f1;
    x2 =x*x; x3 = x*x2; x4 = x*x3; y2 = y*y;
    p1=pow(y-x2+4,2);
    p2=pow(y+x2-4,2);
    p1x = -2*(y-x2+4)*2*x;
    p1y = 2*(y-x2+4);
    p2x = +2*(y+x2-4)*2*x;
    p2y = 2*(y+x2-4);
    f1  = ((1+k1*p1)*(p1x*p2 + p1*p2x)-p1*p2*k1*p1x)/pow(1+k1*p1,2);
    f1  =f1+ k3* ((1+k2*p2)*(p1x*p2 + p1*p2x)-p1*p2*k2*p2x)/pow(1+k2*p2,2);
    f1  =f1+ 2*k4*x;
    return f1;
}

double Upy(double x, double y){

    double x2,x3,x4,y2,p1,p2,p1x,p1y,p2x,p2y,f1,f2;
    x2 =x*x; x3 = x*x2; x4 = x*x3; y2 = y*y;
    p1=pow(y-x2+4,2);
    p2=pow(y+x2-4,2);
    p1x = -2*(y-x2+4)*2*x;
    p1y = 2*(y-x2+4);
    p2x = +2*(y+x2-4)*2*x;
    p2y = 2*(y+x2-4);
    f2  = +((1+k1*p1)*(p1y*p2 + p1*p2y)-p1*p2*k1*p1y)/pow(1+k1*p1,2);
    f2  =f2+k3* ((1+k2*p2)*(p1y*p2 + p1*p2y)-p1*p2*k2*p2y)/pow(1+k2*p2,2);
    return f2;
}

double getg(double x, double y)
{
    double f=((y+x*x-4)*(y+x*x-4));
    double f2=f*f;
    double xi=sqrt(m+f2);
    double den=1/xi+1/M;
    double g=1/den;
    return(g);
}


double getgprime_x(double x,double y)
{
    double f=((y+x*x-4)*(y+x*x-4));
    double fp=4*x*(y+x*x-1);
    double f2=f*f;
    double xi=sqrt(m+f2);
    double num=M*M*f*fp;
    double den=(xi+M)*(xi+M)*xi;
    double res=num/den;
    return(res);
    }

double getgprime_y(double x,double y)
{
    double f=((y+x*x-4)*(y+x*x-4));
    double fp=2*(y+x*x-4);
    double f2=f*f;
    double xi=sqrt(m+f2);
    double num=M*M*f*fp;
    double den=(xi+M)*(xi+M)*xi;
    double res=num/den;
    return(res);
    }

// ///////////////////////////////////////
// /// Bobsled Potential around x=1     //
// ///////////////////////////////////////


// double Upx(double x, double y){

//     double x2,x3,x4,y2,p1,p2,p1x,p1y,p2x,p2y,f1;
//     x2 =x*x; x3 = x*x2; x4 = x*x3; y2 = y*y;
//     p1=pow(y-x2+1,2);
//     p2=pow(y+x2-1,2);
//     p1x = -2*(y-x2+1)*2*x;
//     p1y = 2*(y-x2+1);
//     p2x = +2*(y+x2-1)*2*x;
//     p2y = 2*(y+x2-1);
//     f1  = ((1+k1*p1)*(p1x*p2 + p1*p2x)-p1*p2*k1*p1x)/pow(1+k1*p1,2);
//     f1  =f1+ k3* ((1+k2*p2)*(p1x*p2 + p1*p2x)-p1*p2*k2*p2x)/pow(1+k2*p2,2);
//     f1  =f1+ 2*k4*x;
//     return f1;
// }

// double Upy(double x, double y){

//     double x2,x3,x4,y2,p1,p2,p1x,p1y,p2x,p2y,f1,f2;
//     x2 =x*x; x3 = x*x2; x4 = x*x3; y2 = y*y;
//     p1=pow(y-x2+1,2);
//     p2=pow(y+x2-1,2);
//     p1x = -2*(y-x2+1)*2*x;
//     p1y = 2*(y-x2+1);
//     p2x = +2*(y+x2-1)*2*x;
//     p2y = 2*(y+x2-1);
//     f2  = +((1+k1*p1)*(p1y*p2 + p1*p2y)-p1*p2*k1*p1y)/pow(1+k1*p1,2);
//     f2  =f2+k3* ((1+k2*p2)*(p1y*p2 + p1*p2y)-p1*p2*k2*p2y)/pow(1+k2*p2,2);
//     return f2;
// }

// double getg(double x, double y)
// {
//     double f=((y+x*x-1)*(y+x*x-1));
//     double f2=f*f;
//     double xi=sqrt(m+f2);
//     double den=1/xi+1/M;
//     double g=1/den;
//     return(g);
// }


// double getgprime_x(double x,double y)
// {
//     double f=((y+x*x-1)*(y+x*x-1));
//     double fp=4*x*(y+x*x-1);
//     double f2=f*f;
//     double xi=sqrt(m+f2);
//     double num=M*M*f*fp;
//     double den=(xi+M)*(xi+M)*xi;
//     double res=num/den;
//     return(res);
//     }

// double getgprime_y(double x,double y)
// {
//     double f=((y+x*x-1)*(y+x*x-1));
//     double fp=2*(y+x*x-1);
//     double f2=f*f;
//     double xi=sqrt(m+f2);
//     double num=M*M*f*fp;
//     double den=(xi+M)*(xi+M)*xi;
//     double res=num/den;
//     return(res);
//     }

/////////////////
// Non adaptive one step function //
/////////////////////////////////


int one_step(double ds)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx,j;
    double qy,py,gpy,fy,dwy;
    
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);

    int ns,nt;

    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy) shared(ns,vec_qx,vec_px,vec_qy,vec_py)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // xt::xarray<normal_distribution<double>> dw{ 
        //       normal_distribution<double>{0.0, 1.0 },                                                     
        //       normal_distribution<double>{0.0, 1.0 } };

        // X coordinates
        qx = -1.5;
        px = 1.;

        // Y coordinates
        qy = -0.5;
        py = 1.;

        // Values of dU/dx and dU/dy
        fx = -Upx(qx,qy);  
        fy = -Upy(qx,qy);  

        j=0;
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            px += 0.5*ds*fx;
            // -Y coordinates
            py += 0.5*ds*fy;

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*ds*px;
            // -Y coordinates
            qy += 0.5*ds*py;


            //**********
            //* STEP O *
            //**********
            C = exp(-ds*gamma);
            // -X coordinates
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            // -Y coordinates
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*ds*px;
            // -Y coordinates
            qy += 0.5*ds*py;

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            fx = -Upx(qx,qy);
            px += 0.5*ds*fx;
            // -Y coordinates
            fy = -Upy(qx,qy);
            py += 0.5*ds*fy;

            // To do later 
            if (nt%printskip==0){
            vec_qx[ns][j]=qx;
            vec_px[ns][j]=px;
            vec_qy[ns][j]=qy;
            vec_py[ns][j]=py;
            j=j+1;
            }
        }    vector<double> q_list(int(numruns/printskip),0);

    }

fstream file;
string file_name;
string path=PATH;
for(int nsps = 0; nsps<numsam; nsps++){
    file_name=path+"/vec_noada_x"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    copy(vec_qx[nsps].begin(), vec_qx[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_px[nsps].begin(), vec_px[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_noada_y"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qy[nsps].begin(), vec_qy[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_py[nsps].begin(), vec_py[nsps].end(), out_itr);
    file.close();

    }

return 0;
}



double one_step_tr(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av;
    double qy,py,fy,gpy;
    int ns,nt;
    g_av=0;


    // Savethe values 
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g,g_av)

    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =-1.5;
        px = 1.;

        // Y- coordinates 
        qy = -0.5;
        py = 1.;

        // 
        gpx=getgprime_x(qx,qy);
        gpy=getgprime_y(qx,qy);

        fx = -Upx(qx,qy);   // force
        fy = -Upy(qx,qy);   // force

        // g_av=0.;
        g = getg(qx,qy);
        gdt = dt*g;

        j=0;
        for(nt = 0; nt<numruns; nt++)
        {

                      //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // X- coordinates 
            px += 0.5*gdt*fx;
            // Y- coordinates 
            py += 0.5*gdt*fy;


            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+dt/4*px*g,qy+dt/4*py*g);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            g0=getg(qx+dt/4*px*g1,qy+dt/4*py*g1);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            gdt=g1*dt;
            
            // X- coordinates 
            qx += 0.5*gdt*px;
            // Y- coordinates 
            qy += 0.5*gdt*py;

            //**********
            //* STEP O *
            //**********
            g = getg(qx,qy);
            gdt = dt*g;
            C = exp(-gdt*gamma);
            gpx=getgprime_x(qx,qy);
            gpy=getgprime_y(qx,qy);
            // X- coordinates 
            px = C*px+(1.-C)*tau*gpx/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);
             // Y- coordinates 
            py = C*py+(1.-C)*tau*gpy/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+dt/4*px*g,qy+dt/4*py*g);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            g0=getg(qx+dt/4*px*g1,qy+dt/4*py*g1);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            gdt=g1*dt;
            
            // X- coordinates 
            qx += 0.5*gdt*px;
            // Y- coordinates 
            qy += 0.5*gdt*py;

            //**********
            //* STEP B *
            //**********
            // X- coordinates 
            fx = -Upx(qx,qy);   // force
            fy = -Upy(qx,qy);   // force           
            g = getg(qx,qy);
            gdt = dt*g;

            // X- coordinates 
            px += 0.5*gdt*fx;
            // Y- coordinates 
            py += 0.5*gdt*fy;


            // * Save values of g
            g_av+=g;

            // save the value every %nsnapshot value
            if (nt%printskip==0){
            vec_qx[ns][j]=qx;
            vec_px[ns][j]=px;
            vec_qy[ns][j]=qy;
            vec_py[ns][j]=py;

            vec_g[ns][j]=g;
            j=j+1;
            }


        }
    }
g_av=g_av/(numsam*numruns);

fstream file;
string file_name;
string path=PATH;
for(int nsps = 0; nsps<numsam; nsps++){
    file_name=path+"/vec_tr_x"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    copy(vec_qx[nsps].begin(), vec_qx[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_px[nsps].begin(), vec_px[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_y"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qy[nsps].begin(), vec_qy[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_py[nsps].begin(), vec_py[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_g"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
    file.close();

    }


return g_av;
}


int main(void) {    
    double g_av= one_step_tr();
    cout<<g_av;
    double newds=g_av*dt;
    // //Non adaptive step 
    int out= one_step(newds);


return 0;
}