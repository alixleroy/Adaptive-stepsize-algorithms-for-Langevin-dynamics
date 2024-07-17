/* This code produces results on the accuracy results found in figure 11 and 12. It implements the 
   IP-transformed underdamped Langevin dynamic with the two numerical integration methods. 
   It uses the numerical method: \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} and the method 
   \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} for the 2d potential (6.3)
   To run the code, install fopenmp as per Readme instructions.
   This code contains: 
   - Definition of fixed parameters
   - Define the path and parameters and data can be generated for either figure 11(b) or 12(a) and (b)
   - Definition of the functions: 
        * Upx and Upy: the partial derivatives of the potential defined in (6.3) 
        * getg: the monitor function defined in (6.4)
        * getgprime_x getgprime_y: the partial derivative of the above monitor function 
        * num_int_baoab: numerical integrator method of BAOAB
        * num_int_tilde_baoab: numerical integrator method of \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}
        * main: run the algorithms and save the results of the moments
*/

//
// Include required packages 
//

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
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
using namespace std::chrono;
using namespace std;

//
// Defined fixed parameters 
//

// The parameters of the monitor function are r=1 and alpha=2

#define m               0.2            // minimum step scale factor
#define M               1.             // maximum step scale factor
#define numsam          1              // number of sample
#define T               100000         // total number of trajectories
#define tau             .1            
#define numruns         int(T/dt)       // total number of trajectories
#define gamma           .5              // friction coefficient
#define printskip       10
#define k1              .1
#define k2              50.
#define k3              50.
#define k4              .1

//
// !! Modify the code here depending on which results you want to generate
//
// Use the two following lines to generate data for figure 12 (a) and (b) 
// #define dt              0.025
// #define PATH   "./data/underdamped_2d/smalldt"

// Comment out the above two lines 
// and uncomment those two if you want the results for the figure 11(b)
#define dt              0.005
#define PATH   "./data/underdamped_2d/justnoada"

//
/// A system with two pathways
//


double Upx(double x, double y){
    /*
    Defined the partial derivative with regards to x of the potential presented in (6.3)
    Input
    -----
    x: double 
        value of the position in x 
    y: double 
        value of the position in y
    Return
    ------
    d/dx(U(x,y)): double
        value of the derivative of the potential in x.
    */
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
    /*
    Defined the partial derivative with regards to y of the potential presented in (6.3)
    Input
    -----
    x: double 
        value of the position in x
    y: double 
        value of the position in y
    Return
    ------
    d/dy(U(x,y)): double
        value of the derivative of the potential in x,y.
    */
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
    /*
    Defines the monitor function based on (6.4).
    
    Input
    -----
    x: double 
        value of the position in x 
    y: double 
        value of the position in y 
    Return
    ------
    g(x,y): double
        value of the monitor function in x,y.
    */
    double f=((y+x*x-4)*(y+x*x-4));
    double f2=f*f;
    double xi=sqrt(m+f2);
    double den=1/xi+1/M;
    double g=1/den;
    return(g);
}


double getgprime_x(double x,double y)
{   /*
    Defines the partial derivative in x of the monitor function based on (6.4).
    
    Input
    -----
    x: double 
        value of the position in x 
    y: double 
        value of the position in y 
    Return
    ------
    d/dx g(x,y): double
        value of the partial derivative in x of the monitor function at (x,y).
    */
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
    /*
    Defines the partial derivative in y of the monitor function based on (6.4).
    
    Input
    -----
    x: double 
        value of the position in x 
    y: double 
        value of the position in y 
    Return
    ------
    d/dx g(x,y): double
        value of the partial derivative in y of the monitor function at (x,y).
    */
    double f=((y+x*x-4)*(y+x*x-4));
    double fp=2*(y+x*x-4);
    double f2=f*f;
    double xi=sqrt(m+f2);
    double num=M*M*f*fp;
    double den=(xi+M)*(xi+M)*xi;
    double res=num/den;
    return(res);
    }
//
// Numerical integrator BAOAB for the system (1.1)
//


int num_int_baoab(double ds)
{
    /*
    This function implements the numerical integrator BAOAB, and save the values of the chain 
    of the position and the momentum in a file "data/vec_noada_xi=j.txt" and "data/vec_noada_yi=j.txt"
    where j is the index of the stepsize used to obtain the samples in the list 
    dtlist. This function saves: 
     - "./data/underdamped_2d/vec_noada_xi=j.txt"
     - "./data/underdamped_2d/vec_noada_yi=j.txt"
     which are respectively the values of the momentum and position in the x dimension and the values 
     of the momentum and position in the y dimension. To differentiate between p and q values, the files 
     firslty save "q\n" then the positions values and then "p\n" and the momentum values.
    
    Input
    -----
    ds: double
        value of the stepsize

    Return
    ------
    0: double 
    */
    random_device rd1; //random device generator
    boost::random::mt19937 gen(rd1()); //random device generator
    // set variables type
    double qx,px,gpx,C,fx,dwx,j; 
    double qy,py,gpy,fy,dwy; 
    //empty vector to save values 
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);

    int ns,nt; // set up iterator

    //the following pragma command allow the compiler to run the code in parallel
    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy) shared(ns,vec_qx,vec_px,vec_qy,vec_py)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

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

//
// \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} 
// or BAOAB for the IP-transformed system with correction step in B
//

double num_int_tilde_baoab(void)
{
    /*
    This function implements the numerical integrator \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}, and 
    save the values of the chain of the position, the momentum and monitor function in a file.  
    This function saves:
     - "./data/underdamped_2d/vec_noada_xi=j.txt"
     - "./data/underdamped_2d/vec_noada_yi=j.txt"
     - "./data/underdamped_2d/vec_tr_gi=j.txt"
     which are respectively the values of the momentum, position and monitor function in the x dimension 
     and the values of the momentum and position in the y dimension. To differentiate between p, q and g 
     values, the files firslty save "q\n" then the positions values and then "p\n" and the momentum values
     and finally "g\n" then the values of the monitor function. 

    Input
    -----
    ds: double
        value of the stepsize

    Return
    ------
    0: double 
    */
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
    double g_av= num_int_tilde_baoab();
    cout<<g_av;
    double newds=g_av*dt;
    // //Non adaptive step 
    int out= num_int_baoab(newds);


return 0;
}