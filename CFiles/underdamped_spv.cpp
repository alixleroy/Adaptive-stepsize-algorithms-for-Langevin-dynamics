/* This code produces results on the accuracy results found in figure 8. It implements the 
   IP-transformed underdamped Langevin dynamic with the two numerical integration methods. 
   It uses the numerical method: \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} and the method 
    \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}.
   To run the code, install fopenmp as per Readme instructions.
   This code contains: 
   - Definition of fixed parameters
   - Definition of the functions: 
        * Up: the derivative of the potential defined in (2.1) 
        * getg: the monitor function defined in (3.1)
        * getgprime: the derivative of the above monitor function 
        * num_int_spv: numerical integrator method of spv
        * num_int_tr: numerical integrator method of SPV on IP transformed system of SDE
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
using namespace std::chrono;
using namespace std;


//
// Defined fixed parameters 
//
#define gamma           0.1            // friction coefficient
#define tau             1.            // 'temperature'
#define Tf              40000         // Number of steps forward in time
#define numsam          100 //000       // total number of trajectories
#define printskip       100		// skip this number when saving final values of the vector (should be high as we can't save 10^7 traj) vector
#define printskip2	    100		// use every printskip2 val in a trajectory for the computation of the observable, burnin is 10 000
#define defnburnin      10000   // number of values to skip before saving observable
#define tolA            1e-16
#define nmax            100

//
// Modified harmonic potential parameters 
//

// where we save the moments generated
#define PATH   "./data/underdamped_accuracy_spv"
vector<double> dtlist = {0.202, 0.216, 0.231, 0.247, 0.264, 0.282, 0.301, 0.322, 0.344,0.368};

#define m               .1              //define the lower bound of the monitor function
#define M               1.1             //define the upper bound of the monitor function
#define a               2.75
#define b               0.1
#define x0              0.5
#define c               0.1

//
// Modified harmonic potential parameters 
//

long double Up(double x)
{
    /*
    Defined the derivative of the modified harmonic potential (2.1):
        F(x)=-V'(x) = (\omega(x)^2+c)x,\text{ with } \omega(x) = \frac{b}{\frac{b}{a}+(x-x_0)^2},

    Input
    -----
    x: double 
        value of the position

    Return
    ------
    U'(x): double
        value of the derivative of the potential in x.
    */
   long double xx02= (x-x0)*(x-x0);
   long double wx =b/(b/a+xx02);
    return (wx*wx+c)*x;
}


double getg(double x)
{
    /*
    Defined the monitor function based on (3.1) using the choice of g_3. 
    
    Input
    -----
    x: double 
        value of the position

    Return
    ------
    g(x): double
        value of the monitor function in x.
    */
    double wx,f,xi,g;
    wx =(b/a+pow(x-x0,2.))/b;
    f = wx*wx;
    xi = f+m*m;
    g = 1./(1./M+1./sqrt(xi));
    return(g);

}

double getgprime(double x)
{
    /*
    Defines the derivative of the monitor function based on (3.1) using the choice of g_3. 
    
    Input
    -----
    x: double 
        value of the position

    Return
    ------
    g'(x): double
        value of the derivative of the monitor function in x.
    */
    double wx,f,fp,xi,gprime;
    wx =(b/a+pow(x-x0,2.))/b;
    f = wx*wx;
    fp = 4.*(x-x0)*((b/a)+pow(x-x0,2.))/(b*b);
    xi=sqrt(f+m*m);
    gprime= M*M*fp/(2.*xi*(xi+M)*(xi+M));
    return(gprime);
}

//
// Numerical integrator spv for the system (1.1)
//
vector<double> num_int_spv(double dt, double numruns, double nburnin,int i,int dN)
{
        /*
    This function implements the numerical integrator spv, and save the values of the chain 
    of the position and the momentum in a file "data/vec_noada_qi=j.txt" and "data/vec_noada_pi=j.txt"
    where j is the index of the stepsize used to obtain the samples in the list 
    dtlist. This function saves: 
     - "./data/underdamped_accuracy_spv/vec_noada_qi=j.txt"
     - "./data/underdamped_accuracy_spv/vec_noada_pi=j.txt"
     which are the values of the respectively of a number of numsam/printskip values of the 
     position and the momentum at time T, for the stepsize at index j in the list of stepsize
     dtlist.

     

    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    nburnin: double
        the number of steps to skip before saving values to compute the moments
    i: int
        the index of the stepsize in the list of stepsize dtlist
    dN: int
        the number of values to skip in the chain before saving a value of the 
        sample to compute the moments
    
    Return
    ------
    moments: vector<double> of size (4,1)
        A vector of size 4 where we save the values of the computed first, second
        third and fourth moments. 
    */
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
    vector<double> moments(8,0);
    nsp=0;
    nsp2=0;
    #pragma omp parallel private(q,p,f,C,nt) shared(ns,vec_q,vec_p,moments,nsp,nsp2)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.2;
        p = 0.;
        f = -Up(q);  
        for(nt = 0; nt<numruns; nt++)
        {
                        //
            // SPV integrator
            //
           //**********
            //* STEP X *
            //**********
            q=q+dt*p/2.;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            C = exp(-dt*gamma);
            p = C*p+f*(1.-C)/gamma+sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP X *
            //**********
            q=q+dt*p/2.;

	    if(nt%dN==0 && nt>nburnin){
		moments[0]+=q;
		moments[1]+=q*q;
		moments[2]+=q*q*q;
        moments[3]+=q*q*q*q;
		nsp2+=1;
		}
        }


    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        nsp+=1;
        }
    }

// rescale the moments 
moments[0]=moments[0]/nsp2;
moments[1]=moments[1]/nsp2;
moments[2]=moments[2]/nsp2;
moments[3]=moments[3]/nsp2;

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

return moments;
}

//
// SPV for IP-transformed
//
//
vector<double> num_int_tr(double dt, double numruns,double nburnin, int i,int dN)
{
       /*
    This function implements the numerical integrator SPV for the IP transformed system of SDE, 
    and save the values of the chain of the position, the momentum and monitor function in a file.  
    This function saves: 
     - "./data/underdamped_accuracy_spv/vec_tr_B_qi=j.txt"
     - "./data/underdamped_accuracy_spv/vec_tr_B_pi=j.txt"
     - "./data/underdamped_accuracy_spv/vec_tr_B_gi=j.txt"
     which are the values of the respectively of a number of numsam/printskip values of the 
     position, the momentum and the monitor function g(q) at time T, for the stepsize at index 
     j in the list of stepsize dtlist.
     
     
    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    nburnin: double
        the number of steps to skip before saving values to compute the moments
    i: int
        the index of the stepsize in the list of stepsize dtlist
    dN: int
        the number of values to skip in the chain before saving a value of the 
        sample to compute the moments
    
    Return
    ------
    moments: vector<double> of size (4,1)
        A vector of size 4 where we save the values of the computed first, second
        third and fourth moments. 
    */
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,q0,q1,g_av,g_av_sample,diff;
    int ns,nt,nsp,nsp2,nit;

    // Savethe values 
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    // Initialise snapshot
    nsp=0;
    nsp2=0;
    g_av_sample=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,gp,g,q0,q1,g_av,diff,nit) shared(ns,vec_q,vec_p,vec_g,moments,nsp,nsp2,g_av_sample)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        g_av=0;
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.2;
        p = 0.;
        g = getg(q);
        gdt = dt*g;
        gp=getgprime(q);
        f = -Up(q);   // force
        for(nt = 0; nt<numruns; nt++)
        {
            //**********
            //* STEP Q *
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
            //* STEP P *
            //**********
            f = -Up(q);
            g = getg(q);
            gdt = dt*g;
            gp=getgprime(q);
            C = exp(-gdt*gamma);
            p = C*p+f*(1.-C)/gamma+sqrt((1.-C*C)*tau)*normal(generator)+(1.-C)*tau*gp/(gamma*g) ;

            //**********
            //* STEP Q *
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
            f = -Up(q);
            g = getg(q);
            gdt = dt*g;
            gp=getgprime(q);

    

            //*****************************
            //* Save values of g to average
            //******************************
            g_av+=g;

	    if (nt%dN==0 && nt>nburnin){
		moments[0]+=q;
		moments[1]+=q*q;
		moments[2]+=q*q*q;
		moments[3]+=abs(q-q1);;
		nsp2+=1;	
	        }
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



    // rescale the moments 
    moments[0]=moments[0]/nsp2;
    moments[1]=moments[1]/nsp2;
    moments[2]=moments[2]/nsp2;
    moments[3]=moments[3]/nsp2;

    //*****************************
    //* Save values of g to average
    //******************************
    moments[4]=g_av_sample/numsam; //instead of the moment, we will save the values of the average of g


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
    return moments;
    }





int main(void) {    
 /*
    This function loops through the values of the stepsize in the list of stepsize dtlist
    and run the numerical integrators spv, and SPV onto IP transformed SDE.
    It takes the list of moments generated by each functions and save the moments in files for each numerical integrator.  
    This function saves: 
     - "./data/underdamped_accuracy_spv/noada_moment1.txt"
     - "./data/underdamped_accuracy_spv/noada_moment2.txt"
     - "./data/underdamped_accuracy_spv/noada_moment3.txt"
     - "./data/underdamped_accuracy_spv/noada_moment4.txt"
    which are the values of the first, second, third and fourth moments for the spv method, 
    for each stepsize in the list dtlist. 
     - "./data/underdamped_accuracy_spv/moments_trB_1.txt"
     - "./data/underdamped_accuracy_spv/moments_trB_2.txt"
     - "./data/underdamped_accuracy_spv/moments_trB_3.txt"
     - "./data/underdamped_accuracy_spv/moments_trB_4.txt"
    which are the values of the first, second, third and fourth moments for the modified SPV method 
    applied to the IP-transformed underdamped system.
     - "./data/underdamped_accuracy_spv/vec_g_B.txt"
     which saves the average values taken by the monitor functions accross all samples for each chains. 
     It provides an estimate of the computational effort required to run the numerical methods for 
     the IP-transformed numerical methods. 
     - "./data/underdamped_accuracy_spv/parameters_used.txt"
     provides a list of the parameters used for the simulation as well as the time required to obtain those results. 

    Input
    -----
    void
    
    Return
    ------
    void
    */
    // Compute how much time it takes
    auto start = high_resolution_clock::now();
    using namespace std;
    vector<double> moments_1(dtlist.size(),0);
    vector<double> moments_2(dtlist.size(),0);
    vector<double> moments_3(dtlist.size(),0);
    vector<double> moments_4(dtlist.size(),0);

    vector<double> moments_trB_1(dtlist.size(),0);
    vector<double> moments_trB_2(dtlist.size(),0);
    vector<double> moments_trB_3(dtlist.size(),0);
    vector<double> moments_trB_4(dtlist.size(),0);
    vector<double> vec_g_B(dtlist.size(),0);

    vector<double> moments_trO_1(dtlist.size(),0);
    vector<double> moments_trO_2(dtlist.size(),0);
    vector<double> moments_trO_3(dtlist.size(),0);
    vector<double> moments_trO_4(dtlist.size(),0);
    vector<double> vec_g_O(dtlist.size(),0);

    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples

        double dti = dtlist[i];
        double ni = int(Tf/dti);
        double nburnin = defnburnin;
        double dN=printskip;
       
        // SPV no adaptivity 
        vector<double> moments_di=num_int_spv(dti,ni,nburnin,i,dN);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3];


        // transformed SPV 
        moments_di=num_int_tr(dti,ni,nburnin,i,dN);
        moments_trB_1[i]=moments_di[0];
        moments_trB_2[i]=moments_di[1];
        moments_trB_3[i]=moments_di[2];
        moments_trB_4[i]=moments_di[3];
        vec_g_B[i]=moments_di[4];



       // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path=PATH;

    // NON ADAPTIVE
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/noada_moment1.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(moments_1.begin(), moments_1.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment2.txt";
    file.open(file_name,ios_base::out);
    copy(moments_2.begin(), moments_2.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment3.txt";
    file.open(file_name,ios_base::out);
    copy(moments_3.begin(), moments_3.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment4.txt";
    file.open(file_name,ios_base::out);
    copy(moments_4.begin(), moments_4.end(), out_itr);
    file.close();

    // TRANSFORMED with corr in B 
    file_name=path+"/tr_moment1B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_1.begin(), moments_trB_1.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment2B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_2.begin(), moments_trB_2.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment3B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_3.begin(), moments_trB_3.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment4B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_4.begin(), moments_trB_4.end(), out_itr);
    file.close();

    file_name=path+"/vec_g_B.txt";
    file.open(file_name,ios_base::out);
    copy(vec_g_B.begin(), vec_g_B.end(), out_itr);
    file.close();

    // TRANSFORMED with corr in O 
    file_name=path+"/tr_moment1O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_1.begin(), moments_trO_1.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment2O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_2.begin(), moments_trO_2.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment3O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_3.begin(), moments_trO_3.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment4O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_4.begin(), moments_trO_4.end(), out_itr);
    file.close();

    file_name=path+"/vec_g_O.txt";
    file.open(file_name,ios_base::out);
    copy(vec_g_O.begin(), vec_g_O.end(), out_itr);
    file.close();

    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    string parameters="M="+to_string(M)+"-m="+to_string(m)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-a="+to_string(a)+"-b="+to_string(b)+"-x0="+to_string(x0)+"-c="+to_string(c)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    // string parameters="M1="+to_string(M1)+"-m1="+to_string(m1)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-r="+to_string(r)+"-d="+to_string(d)+"-c="+to_string(c)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
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






