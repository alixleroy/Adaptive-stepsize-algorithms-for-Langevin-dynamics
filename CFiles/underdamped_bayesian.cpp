
/* This code produces results on the accuracy results found in figure 10. It implements the 
   IP-transformed underdamped Langevin dynamic with the two numerical integration methods and the 
   original BAOAB method. 
   It uses the numerical method: \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} and the method 
    \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}.
   It runs the code for the Bayesian model, and compute the number of escaping trajectories.
   To run the code, install fopenmp as per Readme instructions.
   This code contains: 
   - Definition of fixed parameters
   - Definition of the functions: 
        * Up: the derivative of the potential defined in (6.1) 
        * getg: the monitor function defined in (6.2)
        * getgprime: the derivative of the above monitor function 
        * n_escp: numerical integrator method of BAOAB
        * n_escp_tr_B: numerical integrator method of \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}
        * n_escp_tr_O: numerical integrator method of \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}
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
#define Nt              100000         // Number of steps forward in time
#define numsam          1000         // total number of trajectories
#define printskip       1		       // skip this number when saving final values of the vector (should be high as we can't save 10^7 traj) vector
#define printskip2	    100		       // use every printskip2 val in a trajectory for the computation of the observable, burnin is 10 000
#define burnin          1000           // number of values to skip before saving observable
#define tolA            1e-12          // the tolerance on the fixed point integration in step A
#define nmax            100            // the maximum number of steps

//
// Bayesian potential (6.1)  
//

#define PATH   "./data/bayesian_escaping";
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
    /*
    Defined the derivative of the modified harmonic potential (6.1):
        V' (\mu)&= -\left({\sum}_{i=1}^{N} y_i - N \mu - 2 K(\mu-a)^{2 K -1}\right).
    Input
    -----
    x: double 
        value of the position

    Return
    ------
    U'(x): double
        value of the derivative of the potential in x.
    */
    double res =-(sumX-N*x-2.*K*pow(x-a,2.*K-1.));
    return res;
}

double getg(double x)
{
    /*
    Defines the monitor function based on (6.2).
    
    Input
    -----
    x: double 
        value of the position

    Return
    ------
    g(x): double
        value of the monitor function in x.
    */
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
    /*
    Defines the monitor function based on (6.2).
    
    Input
    -----
    x: double 
        value of the position

    Return
    ------
    g(x): double
        value of the monitor function in x.
    */
    double f,f2,xi,fp,gp,sqf; 
    f=pow(x-a-0.5*(sumX/N-a),2.)*r;
    f2=f*f;
    fp=2.*(x-a-0.5*(sumX/N-a))*r; 
    xi=sqrt(1.+m1*f2);
    sqf=sqrt(f2);
    gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
    return(gp);
    }

//
// Numerical integrator BAOAB for the system (1.1)
//

double n_escp(double dt, double numruns, int i)
{
    /*
    This function implements the numerical integrator BAOAB, and save the values of the chain 
    of the position and the momentum in a file "data/vec_noada_qi=j.txt" and "data/vec_noada_pi=j.txt"
    where j is the index of the stepsize used to obtain the samples in the list 
    dtlist. This function saves: 
     - "./data/bayesian_escaping/vec_noada_qi=j.txt"
     - "./data/bayesian_escaping/vec_noada_pi=j.txt"
     which are the values of the respectively of a number of numsam/printskip values of the 
     position and the momentum at time T, for the stepsize at index j in the list of stepsize
     dtlist.

     

    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    i: int
        the index of the stepsize in the list of stepsize dtlist
    Return
    ------
    nsp2: int
        The value of the number of of escaping trajectories. 
    */
    random_device rd1; //tool for random number generation
    boost::random::mt19937 gen(rd1()); //tool for random number generation

    // set up variables type
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp,nsp2;
    // Empty vector to save the values
    vector<double> vec_q(numsam/printskip,0);
    vector<double> vec_p(numsam/printskip,0);
    // Set up iterators
    nsp=0;
    nsp2=0;
    //the following pragma command allow the compiler to run the code in parallel
    #pragma omp parallel private(q,p,f,C,nt) shared(ns,vec_q,vec_p,nsp,nsp2)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        mt19937 generator(rd1()); // Normal generator 
        normal_distribution<double> normal(0, 1); // Normal generator 
        q = 1.23; //initial position
        p = 0.; //initial momentum
        f = -Up(q);   //initial value of the force
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
    // if the value of the position is NaN (escaping trajectory)
    // then add one to the counter of number of escaping trajectory
	if(isnan(q)==true){
	 nsp2+=1;
	}

    // Save every printskip values of the final generated vector    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        nsp+=1;
        }
    }


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

return nsp2;
}

//
// \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} 
// or BAOAB for the IP-transformed system with correction step in B
//
vector<double> n_escp_tr_B(double dt, double numruns, int i)
{
        /*
    This function implements the numerical integrator \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}, and 
    save the values of the chain of the position, the momentum and monitor function in a file.  
    This function saves: 
     - "./data/bayesian_escaping/vec_tr_B_qi=j.txt"
     - "./data/bayesian_escaping/vec_tr_B_pi=j.txt"
     - "./data/bayesian_escaping/vec_tr_B_gi=j.txt"
     which are the values of the respectively of a number of numsam/printskip values of the 
     position, the momentum and the monitor function g(q) at time T, for the stepsize at index 
     j in the list of stepsize dtlist.
     
     
    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    i: int
        the index of the stepsize in the list of stepsize dtlist
    Return
    ------
    nsp2: int
        The value of the number of of escaping trajectories. 
    */
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    
    // Set up variables
    double q,p,f,g,gp,gdt,C,g0,g1,g_av,g_av_sample,q0,q1,diff;
    int ns,nt,nsp,nsp2,nit;

    // empty vector to save samples of the values of the position, momentum and monitor function
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // An empty vector to save the value of average of monitor function 
    // and the count of the number of escaping trajectory.
    vector<double> res_v(2,0);

    // Set up the iterator
    nsp=0;
    nsp2=0;
    g_av_sample=0;

    #pragma omp parallel private(q,p,f,C,nt,gdt,gp,g,g0,g1,g_av,q0,q1,diff,nit) shared(ns,vec_q,vec_p,vec_g,nsp,nsp2,g_av_sample)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        g_av=0;
        mt19937 generator(rd1()); // Normal generator 
        normal_distribution<double> normal(0, 1); // Normal generator 
        q = 1.23; //value of the position
        p = 0.; //value of the momentum
        g = getg(q); //value of the monitor function
        gdt = dt*g; 
        gp=getgprime(q); //valeu of derivative of the monitor function
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
        // compute the number of escaping trajectories
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

    // return the average value of the monitor function 
    // and the number of escaping trajectories
    res_v[0]=nsp2;
    res_v[1]=g_av_sample/numsam;

    return res_v;
    }

//
// \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} 
// or BAOAB for the IP-transformed system with correction step in O
//
vector<double> n_escp_tr_O(double dt, double numruns, int i)
{      /*
    This function implements the numerical integrator \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}, and 
    save the values of the chain of the position, the momentum and monitor function in a file.  
    This function saves: 
     - "./data/bayesian_escaping/vec_tr_O_qi=j.txt"
     - "./data/bayesian_escaping/vec_tr_O_pi=j.txt"
     - "./data/bayesian_escaping/vec_tr_O_gi=j.txt"
     which are the values of the respectively of a number of numsam/printskip values of the 
     position, the momentum and the monitor function g(q) at time T, for the stepsize at index 
     j in the list of stepsize dtlist.
     
     
    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    i: int
        the index of the stepsize in the list of stepsize dtlist
    Return
    ------
    nsp2: int
        The value of the number of of escaping trajectories. 
    */
   // see function n_escp_tr_B for exhaustive code commenting
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,g0,g1,g_av,g_av_sample,diff,q0,q1;
    int ns,nt,nsp,nsp2,nit;

    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);
    // an empty vector to save the value of average of monitor function 
    // and the count of the number of escaping trajectory.
    vector<double> res_v(2,0);

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

        double dti = dtlist[i];
        double ni = Nt;

        // transformed with corr in step O 
        vector<double> res_v =n_escp_tr_O(dti,ni,i);
        moments_trO_1[i]=res_v[0];

        // Plot the average value taken by the monitor function
        cout<<"\n";
        cout<<res_v[1];
        cout<<"\n";

        // transformed with corr in step B 
        res_v =n_escp_tr_B(dti,ni,i);
        moments_trB_1[i]=res_v[0];

        // no adaptivity 
        // rescale by a lower bound on the average value taken by the monitor funciton 
        // to obtain fair comparison
        double res=n_escp(dti*0.85,ni,i);
        moments_1[i]=res;
       



    //
    // save the computed moments
    //
    string path=PATH;

    // Non adaptive
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/noada_nescaping.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(moments_1.begin(), moments_1.end(), out_itr);
    file.close();

    // Transformed with corr in B 
    file_name=path+"/tr_B_nescaping.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_1.begin(), moments_trB_1.end(), out_itr);
    file.close();


    // Transformed with corr in O 
    file_name=path+"/tr_O_nescaping.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_1.begin(), moments_trO_1.end(), out_itr);
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



