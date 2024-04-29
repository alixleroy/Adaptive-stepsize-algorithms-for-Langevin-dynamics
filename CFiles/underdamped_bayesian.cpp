
// Spring potential 
// v2 implies BAOAB with \nabla g computed in step O
// AND fixed point integration for step A



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

using namespace std::chrono;
 
using namespace std;


#define gamma           0.1            // friction coefficient
#define tau             1.            // 'temperature'
#define Nt              100000         // Number of steps forward in time
#define numsam          100000       // total number of trajectories

#define printskip       1		// skip this number when saving final values of the vector (should be high as we can't save 10^7 traj) vector
#define printskip2	    100		// use every printskip2 val in a trajectory for the computation of the observable, burnin is 10 000
#define burnin          1000   // number of values to skip before saving observable
#define tolA            0.000000001
#define nmax            50
///////////////////// DEFINE POTENTIAL //////////////////////////////





/////////////////////////////////////////////////////////////
/////////////////// BAYESIAN //////////////////////////////
/////////////////////////////////////////////////////////////
//#define PATH "./pot_ben"
#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/bayesian_plots";

// vector<double> dtlist ={0.5   , 0.4789, 0.4579, 0.4368, 0.4158, 0.3947, 0.3737, 0.3526,
//        0.3316, 0.3105, 0.2895, 0.2684, 0.2474, 0.2263, 0.2053, 0.1842,
//        0.1632, 0.1421, 0.1211, 0.1};

vector<double> dtlist = {0.5  , 0.474, 0.448, 0.423, 0.397, 0.371, 0.345, 0.319, 0.294,
       0.268, 0.242, 0.216, 0.191, 0.165, 0.139, 0.113, 0.087, 0.062,
       0.036, 0.01};

#define K     4.
#define a     2.
#define sumX  17.2
#define N     10.
#define r     2.
#define m1    0.1
#define M1    1./1.

double Up(double x)
{
    double res =-(sumX-N*x-2.*K*pow(x-a,2.*K-1.));
    return res;
}

double getg(double x)
{
    double f,f2,xi,den,g;
    f=pow(x-a-0.5*(sumX/N-a),2.)*r;
    f2=f*f;
    xi=sqrt(1.+m1*f2);
    den=M1*xi+sqrt(f2);
    g=xi/den;
    return(g);
}

double getgprime(double x)
{
    double f,f2,xi,fp,gp,sqf; 
    f=pow(x-a-0.5*(sumX/N-a),2.)*r;
    f2=f*f;
    fp=2.*(x-a-0.5*(sumX/N-a))*r; 
    xi=sqrt(1.+m1*f2);
    sqf=sqrt(f2);
    gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
    return(gp);
    }



/////////////////////////////////
// Non adaptive one step function //
/////////////////////////////////

double one_step(double dt, double numruns, int i)
{
    //tools for sampling random increments
    random_device rd1;
    boost::random::mt19937 gen(rd1());

    // set variables
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp,nsp2;
    // Save the values 
    vector<double> vec_q(numsam/printskip,0);
    vector<double> vec_p(numsam/printskip,0);

    // Compute the moments, so its done
    nsp=0;
    nsp2=0;
    #pragma omp parallel private(q,p,f,C,nt) shared(ns,vec_q,vec_p,nsp,nsp2)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 1.23;
        p = 0.;
        f = -Up(q);  
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*dt*f;

            //**********
            //* STEP A *
            //**********
            q += 0.5*dt*p;

            //**********
            //* STEP O *
            //**********
            C = exp(-dt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q += 0.5*dt*p;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            p += 0.5*dt*f;

        }

	if(isnan(q)==true){
	 nsp2+=1;
	}

    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        nsp+=1;
        }
    }

// rescale the moments 

// save the some of the values generated. 
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_noada_q"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();

file_name=path+"/vec_noada_p"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

return nsp2;
}

////////////////////////////////////////////////////////
////////// ADAPTIVE WITH ADAPTIVE STEP IN B ////////////
////////////////////////////////////////////////////////

vector<double> one_step_tr_B(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,g0,g1,g_av,g_av_sample,q0,q1,diff;
    int ns,nt,nsp,nsp2,nit;

    // Savethe values 
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // save the value of g and the value of number of escaping traj
    vector<double> res_v(2,0);


    // Initialise snapshot
    nsp=0;
    nsp2=0;
    g_av_sample=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,gp,g,g0,g1,g_av,q0,q1,diff,nit) shared(ns,vec_q,vec_p,vec_g,nsp,nsp2,g_av_sample)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        g_av=0;
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 1.23;
        p = 0.;
        g = getg(q);
        gdt = dt*g;
        gp=getgprime(q);
        f = -Up(q);   // force
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

            //**********
            //* STEP A *
            //**********
            q0=q;
            q1=q0+dt/2.*p*getg(q0);
            diff=1.;
            nit=0;
            while (nit<nmax && diff>tolA){
                nit+=1;
                q1=q0+dt/2.*p*getg((q0+q)/2.);
                diff=abs(q-q1);
                q=q1;
            }

            //**********
            //* STEP O *
            //**********
            g = getg(q);
            gdt = dt*g;
            C = exp(-gdt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q0=q;
            q1=q0+dt/2.*p*getg(q0);
            diff=1.;
            nit=0;
            while (nit<nmax && diff>tolA){
                nit+=1;
                q1=q0+dt/2.*p*getg((q0+q)/2.);
                diff=abs(q-q1);
                q=q1;
            }

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            g = getg(q);
            gp=getgprime(q);
            gdt = dt*g;
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

            //*****************************
            //* Save values of g to average
            //******************************
            g_av+=g;
        }
    
        if (isnan(q)==true){
		nsp2+=1;	
	    }

    //*****************************
    //* Save values of g to average
    //******************************
    g_av=g_av/numruns;
    g_av_sample+=g_av;
    


    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        vec_g[nsp]=g;
        nsp+=1;
        }
    
    }


    // save the some of the values generated. 
    string path=PATH;
    fstream file;
    file << fixed << setprecision(16) << endl;
    string list_para="i="+to_string(i); 
    string file_name=path+"/vec_tr_B_q"+list_para+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_q.begin(), vec_q.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_B_p"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_p.begin(), vec_p.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_B_g"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_g.begin(), vec_g.end(), out_itr);
    file.close();

    // return the saved moments
    res_v[0]=nsp2;
    res_v[1]=g_av_sample/numsam;

    return res_v;
    }

////////////////////////////////////////////////////////
////////// ADAPTIVE WITH ADAPTIVE STEP IN O ////////////
////////////////////////////////////////////////////////


vector<double> one_step_tr_O(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,g0,g1,g_av,g_av_sample,diff,q0,q1;
    int ns,nt,nsp,nsp2,nit;

    // Savethe values 
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // Return the save moments
    vector<double> res_v(2,0);


    // Initialise snapshot
    nsp=0;
    nsp2=0;
    g_av_sample=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,gp,g,g0,g1,g_av,q0,q1,diff,nit) shared(ns,vec_q,vec_p,vec_g,nsp,nsp2)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        g_av=0;
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 1.23;
        p = 0.;
        g = getg(q);
        gdt = dt*g;
        gp=getgprime(q);
        f = -Up(q);   // force
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*gdt*f;

            //**********
            //* STEP A *
            //**********
            q0=q;
            q1=q0+dt/2.*p*getg(q0);
            diff=1.;
            nit=0;
            while (nit<nmax && diff>tolA){
                nit+=1;
                q1=q0+dt/2.*p*getg((q0+q)/2.);
                diff=abs(q-q1);
                q=q1;
            }

            //**********
            //* STEP O *
            //**********
            g = getg(q);
            gdt = dt*g;
            C = exp(-gdt*gamma);
            gp=getgprime(q);
            p = C*p+(1.-C)*tau*gp/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q0=q;
            q1=q0+dt/2.*p*getg(q0);
            diff=1.;
            nit=0;
            while (nit<nmax && diff>tolA){
                nit+=1;
                q1=q0+dt/2.*p*getg((q0+q)/2.);
                diff=abs(q-q1);
                q=q1;
            }

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            g = getg(q);
            gdt = dt*g;
            p += 0.5*gdt*f;

            //*****************************
            //* Save values of g to average
            //******************************
            g_av+=g;

        }

    if(isnan(q)==true){
		nsp2+=1;
        }

    //*****************************
    //* Save values of g to average
    //******************************
    g_av=g_av/numruns;
    g_av_sample+=g_av;
    
    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        vec_g[nsp]=g;
        nsp+=1;
        }
    
    }



    // save the some of the values generated. 
    string path=PATH;
    fstream file;
    file << fixed << setprecision(16) << endl;
    string list_para="i="+to_string(i); 
    string file_name=path+"/vec_tr_O_q"+list_para+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_q.begin(), vec_q.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_O_p"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_p.begin(), vec_p.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_O_g"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_g.begin(), vec_g.end(), out_itr);
    file.close();

    // return the saved moments
    res_v[0]=nsp2;
    res_v[1]=g_av_sample/numsam; 


    return res_v;
    }



int main(void) {    

    // Compute how much time it takes
    auto start = high_resolution_clock::now();
    using namespace std;
    vector<double> moments_1(dtlist.size(),0);


    vector<double> moments_trB_1(dtlist.size(),0);


    vector<double> moments_trO_1(dtlist.size(),0);


    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples
        cout<<"\n";
        cout<<1;
        cout<<"\n";
        double dti = dtlist[i];
        double ni = Nt;

        // transformed with corr in step O 
        vector<double> res_v =one_step_tr_O(dti,ni,i);
        moments_trO_1[i]=res_v[0];

        cout<<"\n";
        cout<<res_v[1];
        cout<<"\n";

        // transformed with corr in step B 
        res_v =one_step_tr_B(dti,ni,i);
        moments_trB_1[i]=res_v[0];

        // no adaptivity 
    
        double res=one_step(dti*0.85,ni,i);
        moments_1[i]=res;
       



       // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path=PATH;

    // NON ADAPTIVE
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/noada_nescaping.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(moments_1.begin(), moments_1.end(), out_itr);
    file.close();


    // TRANSFORMED with corr in B 
    file_name=path+"/tr_B_nescaping.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_1.begin(), moments_trB_1.end(), out_itr);
    file.close();


    // TRANSFORMED with corr in O 
    file_name=path+"/tr_O_nescaping.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_1.begin(), moments_trO_1.end(), out_itr);
    file.close();


    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    // string parameters="M="+to_string(M)+"-m="+to_string(m)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-a="+to_string(a)+"-b="+to_string(b)+"-x0="+to_string(x0)+"-c="+to_string(c)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    // string parameters="M1="+to_string(M1)+"-m1="+to_string(m1)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-r="+to_string(r)+"-d="+to_string(d)+"-c="+to_string(c)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string parameters="M1="+to_string(M1)+"-m1="+to_string(m1)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-K="+to_string(K)+"-a="+to_string(a)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();
    }


return 0;
}













// double getg(double x)
// {
//     double f,f2,xi,den,g;
//     f=(r*(x*x*x+d*cos(d*x+1))*(x*x*x+d*cos(d*x+1)));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+abs(f);
//     g=xi/den;
//     return(g);

// }

// double getgprime(double x)
// {
//     double f,f2,fp,xi,gp,signf;
//     // if (x==0){ ;}
//     // else{
//     f=r*(x*x*x+d*cos(d*x+1))*(x*x*x+d*cos(d*x+1));
//     f2=f*f;
//     fp=r*2*(x*x*x+d*cos(d*x+1))*(3*x*x-d*d*sin(1+d*x));
//     xi=sqrt(1+m1*f2);
//     signf= (f > 0) - (f < 0);
//     gp=-signf*fp/(xi*pow(M1*xi+abs(f),2));
//     return(gp);
// }



/////////////////////// DEFINE POTENTIAL //////////////////////////////
// #define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/spring_validate/v2"
// #define PATH "./spring_v2_gamma1"
// vector<double> dtlist ={0.1   , 0.1444, 0.1889, 0.2333, 0.2778, 0.3222, 0.3667, 0.4111,
//        0.4556, 0.5   };

// vector<double> dtlist ={0.001, 0.012, 0.023, 0.034, 0.045, 0.056, 0.067, 0.078, 0.089,0.1 };
// vector<double> dtlist ={0.05  , 0.1222, 0.1944, 0.2667, 0.3389, 0.4111, 0.4833, 0.5556,
//        0.6278, 0.7 };
// /////////////////////////////////
// // Spring potential definition //
// /////////////////////////////////
// #define m               0.001
// #define M               1.1
// // Spring potential 
// //parameters of the potential 
// #define a               15.0
// #define b               0.1
// #define x0              0.5
// #define c               0.1

// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }

// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);

// }

// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }



// /////////////////////////////////
// // Double well
// /////////////////////////////////
// // vector<double> dtlist = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 };  // //vector<double> dtlist = {0.001 , 0.0039, 0.005 , 0.0065, 0.0084, 0.0109, 0.014 , 0.0181,0.0234, 0.0302};
// vector<double> dtlist ={0.05, 0.06, 0.07, 0.08, 0.09, 0.1};

// #define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/dw/v2"

// #define a 1. // parameter of how steep the double well is 
// #define c 0.6 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 
// #define m1    0.1
// #define M1     1/2.

// double U(double x){
//     return ((x+a)*(x+a)-0.0001)*pow((x-c),4);
//     }

// double Up(double x){
//     double xc =x-c;
//     double xa=x+a;
//     double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.0002);
//     return v;}


// // g depends on (x-c)^3
// /////////////////////////////////

// double getg(double x)
// {
//     double f,f2,xi,den,g;
//     f=pow(x-c,3);
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+abs(f);
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     f=pow(x-c,3);
//     f2=f*f;
//     fp=3*pow(x-c,2);
//     xi=sqrt(1+m1*f2);
//     gp=-f*fp/(sqrt(f2)*xi*pow(M1*xi+abs(f),2));
//     return(gp);
//     }

// // g depends on (x-c)^32(x+a)^2
// /////////////////////////////////


// double getg(double x)
// {
//     double f,f2,xi,den,g;
//     f=pow(x-c,3)*2*pow(x+a,2)*low;
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+abs(f);
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     f=low*pow(x-c,3)*2*pow(x+a,2);
//     f2=f*f;
//     fp=low*(x+a)*pow(x-c,2)*(3*a-2*c+5*x)*2;
//     xi=sqrt(1+m1*f2);
//     gp=-f*fp/(sqrt(f2)*xi*pow(M1*xi+abs(f),2));
//     return(gp);
//     }





// /////////////////////////////////
// // Double well 


// #define a 0.4
// #define c 1.6


// long double Up(double x)
// {
//     double xc =x-c;
//     double xa=x+a;
//     double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.0002);
//     return v;
// }

// ///////////////////////////////////////////////////////////////
// // with the definition 1/(1/M+1/f+m) with f=1/((x-a)^2(x+c)^3)
// ///////////////////////////////////////////////////////////////
// double getg(double x)
// {
//     double xc,xa,f,f2,xi,den,g;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*pow(xa,2));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+f;
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*pow(xa,2));
//     f2=f*f;
//     fp=0.5*pow(xa,3)*pow(xc,5)*(3*a-2*c+5*x)/abs(f);
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }







// /////////////////////////////////
// // Spring potential definition //
// /////////////////////////////////
// // Spring potential 
// //parameters of the potential 
// #define a               20.0
// #define b               0.1
// #define x0              0.1
// #define c               0.1

// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }

// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);

// }

// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }

/////////////////////////////////
// Square definition //
/////////////////////////////////


// long double Up(double x)
// {
//    return x;
// }

// double getg(double x)
// {
//     return 1;
// }

// double getgprime(double x)
// {
//     return 0;
// }



/////////////////////////////////
// Spring potential definition //
/////////////////////////////////
// Spring potential 
//parameters of the potential 
// #define a               1.0
// #define b               1.0
// #define x0              0.5
// #define c               0.1
// vector<double> dtlist = {exp(-4.5),exp(-4.21),exp(-3.93),exp(-3.64),exp(-3.36),exp(-3.07),exp(-2.79),exp(-2.5),exp(-2.21),exp(-1.93),exp(-1.64),exp(-1.36),exp(-1.07), exp(-0.79), exp(-0.5)};

// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }

// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);

// }

// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }


// ////////////////////////////////////////////////////////////////////////////////////////////
// ////////////// POTENTIAL x^4+sin(1+dx)
// ///////////////////////////////////////////////////////////////

// // #define PATH "./validate/spring_v2"
// #define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/spring_validate/v2";

// vector<double> dtlist = {0.37,0.3456,0.3211,0.2967,0.2722,0.2478,0.2233,0.1989,0.1744,0.15};
// #define r     0.05
// #define d     5.
// #define m1    .001
// #define M1     1/1.
// #define c       0.5

// double Up(double x)
// {
//     double res = x*x*x+d*cos(1+d*x);
//     return res;
// }


// double getg(double x)
// {
//     double f,f2,xi,den,g;
//     f=r*pow(x,4);
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+abs(f);
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double f,f2,xi,fp,gp,signf;
//     f=r*pow(x,4);
//     f2=f*f;
//     fp=4*x*x*x*r;
//     xi=sqrt(1+m1*f2);
//     signf= (f > 0) - (f < 0);
//     gp=-signf*fp/(xi*pow(M1*xi+abs(f),2));
//     return(gp);
//     }


// /////////////////////////////////
// // Spring potential definition //
// /////////////////////////////////
// #define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/spring_validate/v1"
// // #define PATH "./spring_a4_gamma01"
// // FOR A=4
// vector<double> dtlist ={0.7 , 0.63, 0.56,0.48,0.41,0.34,0.27, 0.19, 0.12, 0.05};

// // FOR A =15
// //larger range of values 
// //vector<double> dtlist = {0.1  , 0.14 , 0.195,0.273, 0.38 , 0.531, 0.741, 1.034, 1.443, 2.014};
// //vector<double> dtlist = {2.014,1.443,1.034,0.741,0.531,0.38,0.273,0.195,0.14,0.1};

// #define m               0.001
// #define M               1.1
// // Spring potential -- parameters of the potential 
// #define a               2.75
// #define b               0.1
// #define x0              0.5
// #define c               0.1

// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }

// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m*m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);

// }

// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }
