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
        * num_int_baoab: numerical integrator method of BAOAB
        * num_int_hat_baoab: numerical integrator method of \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}
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
using namespace std::chrono;
using namespace std;

//
// Defined fixed parameters 
//

#define gamma           0.1            // friction coefficient
#define tau             1.             // 'temperature'
#define Tf              1000          // Number of steps forward in time
#define numsam          1000         // total number of trajectories
#define printskip       100		       // skip this number when saving final values of the vector (should be high as we can't save 10^7 traj) vector
#define printskip2	    100		       // use every printskip2 val in a trajectory for the computation of the observable, burnin is 10 000
#define defnburnin      100          // number of values to skip before saving observable
#define tolA            1e-16          // the tolerance on the fixed point integration in step A
#define nmax            100            // the maximum number of steps

//
// Modified harmonic potential parameters 
//

// where we save the moments generated
#define PATH   "./data/underdamped_accuracy_baoab"
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
// Numerical integrator BAOAB for the system (1.1)
//
vector<double> num_int_baoab(double dt, double numruns, double nburnin,int i,int dN)
{
    /*
    This function implements the numerical integrator BAOAB, and save the values of the chain 
    of the position and the momentum in a file "data/vec_noada_qi=j.txt" and "data/vec_noada_pi=j.txt"
    where j is the index of the stepsize used to obtain the samples in the list 
    dtlist. This function saves: 
     - "./data/underdamped_accuracy_baoab/vec_noada_qi=j.txt"
     - "./data/underdamped_accuracy_baoab/vec_noada_pi=j.txt"
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
    random_device rd1;     //tools for sampling random increments
    boost::random::mt19937 gen(rd1());     //tools for sampling random increments
    
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp,nsp2;
    
    vector<double> vec_q(numsam/printskip,0); // empty vector to save the values of the positions
    vector<double> vec_p(numsam/printskip,0); // empty vector to save the values of the momentum
    vector<double> moments(8,0); // empty vector to save the values of the moments
    
    nsp=0; // initialise the value of the iterator to save every printskip samples for momentum and position
    nsp2=0; //initialise the value of the iterator to count the number of values used to get the moments, as it is necessary to normalise the moments afterwards.
    
    //the following pragma command allow the compiler to run the code in parralel
    #pragma omp parallel private(q,p,f,C,nt) shared(ns,vec_q,vec_p,moments,nsp,nsp2)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){ // loop through the number of samples
        mt19937 generator(rd1()); // Normal generator 
        normal_distribution<double> normal(0, 1); // Normal generator 
        q = 0.2; // initialise position
        p = 0.; // initialise momentum
        f = -Up(q); // compute the force
        for(nt = 0; nt<numruns; nt++) // loop over the number of steps required to reach T
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

        // save the moments using values generated after nburnin steps, and using every dN values
	    if(nt%dN==0 && nt>nburnin){
		moments[0]+=q;
		moments[1]+=q*q;
		moments[2]+=q*q*q;
        moments[3]+=q*q*q*q;
		nsp2+=1;
		}
        }
    // Save values for momentum and position using only samples at time T and every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        nsp+=1;
        }
    }

// Rescale the moments
moments[0]=moments[0]/nsp2;
moments[1]=moments[1]/nsp2;
moments[2]=moments[2]/nsp2;
moments[3]=moments[3]/nsp2;

// save the values of the position and momentum in txt files
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
// \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} 
// or BAOAB for the IP-transformed system with correction step in B
//
vector<double> num_int_hat_baoab(double dt, double numruns,double nburnin, int i,int dN)
{
    /*
    This function implements the numerical integrator \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}, and 
    save the values of the chain of the position, the momentum and monitor function in a file.  
    This function saves: 
     - "./data/underdamped_accuracy_baoab/vec_tr_B_qi=j.txt"
     - "./data/underdamped_accuracy_baoab/vec_tr_B_pi=j.txt"
     - "./data/underdamped_accuracy_baoab/vec_tr_B_gi=j.txt"
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
    random_device rd1;  
    boost::random::mt19937 gen(rd1());
    
    double q,p,f,g,gp,gdt,C,q0,q1,g_av,g_av_sample,diff;
    int ns,nt,nsp,nsp2,nit;
    
    vector<double> vec_q((numsam/printskip),0);  // empty vector to save the values of the position 
    vector<double> vec_p((numsam/printskip),0);  // empty vector to save the values of the momentum
    vector<double> vec_g((numsam/printskip),0);  // empty vector to save the values of the monitor function
    vector<double> moments(8,0); // empty vector to save the values of the moments
    
    nsp=0; // Initialise iterator for the values of the position, momentum and monitor function
    nsp2=0; // intialise iterator to count the number of values used to compute the moments. 
    g_av_sample=0; //intialise an empty variable to obtain an average value of the monitor function
    
    #pragma omp parallel private(q,p,f,C,nt,gdt,gp,g,q0,q1,g_av,diff,nit) shared(ns,vec_q,vec_p,vec_g,moments,nsp,nsp2,g_av_sample)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        g_av=0;

        mt19937 generator(rd1());// Normal generator  
        normal_distribution<double> normal(0, 1);
        q = 0.2; //initialise value of q
        p = 0.; //initial value of p 
        g = getg(q); //initial value of the monitor function
        gdt = dt*g; //value of g(q) at initial position
        gp=getgprime(q); //value of g'(q) at initial position
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

            // compute values of moments using every dN values of q, after nburnin steps
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
    

    // Save the values of the momentum, position and monitor function every printskip values    
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

//
// \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} 
// or BAOAB for the IP-transformed system with correction step in O
//
vector<double> num_int_tilde_baoab(double dt, double numruns,double nburnin, int i,int dN)
{
    /*
    This function implements the numerical integrator \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}, and 
    save the values of the chain of the position, the momentum and monitor function in a file.  
    This function saves: 
     - "./data/underdamped_accuracy_baoab/vec_tr_O_qi=j.txt"
     - "./data/underdamped_accuracy_baoab/vec_tr_O_pi=j.txt"
     - "./data/underdamped_accuracy_baoab/vec_tr_O_gi=j.txt"
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
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,q0,q1,g_av,g_av_sample,diff;
    int ns,nt,nsp,nsp2,nit;

    // Save the values 
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    // Initialise snapshot
    nsp=0;
    nsp2=0;
    g_av_sample=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,gp,g,q0,q1,g_av,diff,nit) shared(ns,vec_q,vec_p,vec_g,moments,nsp,nsp2)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        g_av=0;
        mt19937 generator(rd1()); // Normal generator 
        normal_distribution<double> normal(0, 1); // Normal generator 
        q = 0.2; // initial position
        p = 0.;  // initial momentum
        g = getg(q); // value of the monitor function at initial position
        gdt = dt*g; 
        gp=getgprime(q); //value of the derivative of the monitor function at initial position
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
            gp=getgprime(q);
            C = exp(-gdt*gamma);
            p = C*p+(1.-C)*tau*gp/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q0=q;
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
         	
	    if(nt%dN==0 && nt>nburnin){
		moments[0]+=q;
		moments[1]+=q*q;
		moments[2]+=q*q*q;
		moments[3]+=abs(q1-q);
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
    return moments;
    }

//
// Main
//

int main(void) {    
    /*
    This function loops through the values of the stepsize in the list of stepsize dtlist
    and run the numerical integrators BAOAB, \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} 
    and \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}. It takes the list of moments generated
    by each functions and save the moments in files for each numerical integrator.  
    This function saves: 
     - "./data/underdamped_accuracy_baoab/noada_moment1.txt"
     - "./data/underdamped_accuracy_baoab/noada_moment2.txt"
     - "./data/underdamped_accuracy_baoab/noada_moment3.txt"
     - "./data/underdamped_accuracy_baoab/noada_moment4.txt"
    which are the values of the first, second, third and fourth moments for the BAOAB method, 
    for each stepsize in the list dtlist. 
     - "./data/underdamped_accuracy_baoab/moments_trB_1.txt"
     - "./data/underdamped_accuracy_baoab/moments_trB_2.txt"
     - "./data/underdamped_accuracy_baoab/moments_trB_3.txt"
     - "./data/underdamped_accuracy_baoab/moments_trB_4.txt"
    which are the values of the first, second, third and fourth moments for the \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}
    method, for each stepsize in the list dtlist. This is the numerical integrator which integrates the computation
    of the extra correction term in step B. 
     - "./data/underdamped_accuracy_baoab/moments_trO_1.txt"
     - "./data/underdamped_accuracy_baoab/moments_trO_2.txt"
     - "./data/underdamped_accuracy_baoab/moments_trO_3.txt"
     - "./data/underdamped_accuracy_baoab/moments_trO_4.txt"
    which are the values of the first, second, third and fourth moments for the \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} 
    method, for each stepsize in the list dtlist. This is the numerical integrator which integrates the computation
    of the extra correction term in step O. 
     - "./data/underdamped_accuracy_baoab/vec_g_B.txt"
     - "./data/underdamped_accuracy_baoab/vec_g_O.txt"
     which saves the average values taken by the monitor functions accross all samples for each chains. It provides an estimate 
     of the computational effort required to run the numerical methods for the IP-transformed numerical methods. 
     - "./data/underdamped_accuracy_baoab/parameters_used.txt"
     provides a list of the parameters used for the simulation as well as the time required to obtain those results. 

    Input
    -----
    void
    
    Return
    ------
    void
    */
    auto start = high_resolution_clock::now(); // start the clock so we can compute how long the simulation is
    using namespace std;
    // create empty vector to save the moments for the three simulations
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

        // select the value of h in the list of stepsize
        double dti = dtlist[i];
        double ni = int(Tf/dti);
        double nburnin = defnburnin;
        double dN=printskip;
       
        // BAOAB
        vector<double> moments_di=num_int_baoab(dti,ni,nburnin,i,dN);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3];


        // \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} 
        moments_di=num_int_hat_baoab(dti,ni,nburnin,i,dN);
        moments_trB_1[i]=moments_di[0];
        moments_trB_2[i]=moments_di[1];
        moments_trB_3[i]=moments_di[2];
        moments_trB_4[i]=moments_di[3];
        vec_g_B[i]=moments_di[4];


        // \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} 
        moments_di=num_int_tilde_baoab(dti,ni,nburnin,i,dN);
        moments_trO_1[i]=moments_di[0];
        moments_trO_2[i]=moments_di[1];
        moments_trO_3[i]=moments_di[2];
        moments_trO_4[i]=moments_di[3];
        vec_g_O[i]=moments_di[4];
        
    //
    // save the computed moments
    //
    string path=PATH;

    // BAOAB
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

    // \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}  
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

    // \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} 
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

    //
    // Save the values of the parameters 
    //

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




