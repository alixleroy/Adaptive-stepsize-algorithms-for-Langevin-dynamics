/* This code produces results on the number of iterations required in the fixed step size 
   integration in the step A of the BAOAB numerical integration methods adapted to the 
   IP-transformed underdamped Langevin dynamic.
   To run the code, install fopenmp
   This code contains: 
   - Definition of fixed parameters
   - Definition of the functions: 
        * Up: the derivative of the potential defined in (2.1) 
        * getg: the monitor function defined in (3.1)
        * getgprime: the derivative of the above monitor function 
        * getstepA: compute the fixed point integration with step A
        * num_int_baoab: numerical integrator method of BAOAB
        * num_int_hat_baoab: numerical integrator method of \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}
        * num_int_tilde_baoab: numerical integrator method of \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}
        * main: run the algorithms and save the results
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

#define gamma           0.1                     // friction coefficient
#define tau             1.                      // temperature
#define Nt              100000                  // number of steps forward in time
#define numsam          1                       // total number of trajectories
#define printskip       1		                // skip this number when saving final values of the vector 
#define printskip2	    1		                // use every printskip2 val in a trajectory for the computation of the observable
#define burnin          1000                    // number of values to skip before saving observable
#define tolA            1e-12 //0.000000000001          // tolerance accepted for the difference 
                                                // between the last two iterations of the fixed step size A. 
#define nmax            100                     // the maximum number of iterations in the fixed point integrations 

//
// Modified harmonic potential parameters 
//

#define PATH   "./data/stepA"
#define h               0.1                     // define the stepsize
#define m               0.001                    // lower bound for the monitor function
#define M               1.1                     // upper bound for the monitor function  
#define a               2.75                    // parameters of the modified harmonic potential
#define b               0.1                     
#define x0              0.5                      
#define c               0.1

//
// Modified harmonic potential parameters 
//

double Up(double x)
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
   double xx02= (x-x0)*(x-x0);
   double wx =b/(b/a+xx02);
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
    Defined the derivative of the monitor function based on (3.1) using the choice of g_3. 
    
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


vector<double> getstepA(double q, double p, double dt){
    /*
    Numerical integration of the step A, with fixed step size integration following (5.8) and (5.9)
    Input
    -----
    q: double 
        value of the position
    p: double 
        value of the momentum
    dt: double
        value of the stepsize

    Return
    ------
    resA: vector of size (3,1)
        The vector contains the value of the position, the number of iterations required to 
        reach this value and the difference between the last two iterations.
    */

    double q0,q1k,q1,diff,nit; 
    vector<double> resA(3,0); 
    nit=0;
    q0=q;
    q1k=q0+dt/2.*p*getg(q0); // initial guess for k-1
    diff=1.0; // set a different that is larger than the tolerance
    while(diff>tolA &&  nit<nmax){ //loop until the tolerance is reached or we have reached the maximum number of iterations
        nit+=1; //increment the number of iterations
        q1=q0+dt/2.*p*getg((q0+q1k)/2.); //fixed point integration as in (5.8)
        diff = abs(q1-q1k); // the difference between the current and previous values of q through the iteration
        q1k=q1; 
    }
    resA[0]=q1; // the final value of the position
    resA[1]=nit; // the number of iteration
    resA[2]=diff; // the difference between the last two iteration
    return(resA);
    }

//
// Numerical integrator BAOAB for the system (1.1)
//

int num_int_baoab(double dt, double numruns)
{
    /*
    This function implements the numerical integrator BAOAB, and save the values of the chain 
    of the position and the momentum in a file "data/vec_noada_q.txt" and "data/vec_noada_p.txt"
    
    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    Return
    ------
    0: int
    */

    random_device rd1; // tools for sampling random increments
    boost::random::mt19937 gen(rd1()); // tools for sampling random increments
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;
    vector<double> vec_q(numruns/printskip,0); // empty vector to save the values of the positions
    vector<double> vec_p(numruns/printskip,0); // empty vector to save the values of the momentum
    nsp=0; // initialise the values of the iterator
    //the following pragma command allow the compiler to run the code in parralel
    #pragma omp parallel private(q,p,f,C,nt) shared(ns,vec_q,vec_p,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){ //loop on the number of samples
        mt19937 generator(rd1()); // normal generator
        normal_distribution<double> normal(0, 1); // normal generator with mean 0 and standard deviation of 1
        q = 1.23; // initial position
        p = 0.; //initial momentum
        f = -Up(q); //compute the force
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

            if (nt%printskip==0){
                vec_q[nsp]=q;
                vec_p[nsp]=p;
                nsp+=1;
            }

        }

    }

// save the values in txt files
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;

string file_name=path+"/vec_noada_q.txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();

file_name=path+"/vec_noada_p.txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

return 0;
}

//
// \hat{B}\hat{A}\hat{O}\hat{A}\hat{B} 
// or BAOAB for the IP-transformed system with correction step in B
//
double num_int_hat_baoab(double dt, double numruns)
{
    /*
    This function implements the numerical integrator \hat{B}\hat{A}\hat{O}\hat{A}\hat{B}, and save the 
    values generated under: 
        - "data/vec_noada_B_q.txt" containing the values of the position in the chain for a sample
        - "data/vec_noada_B_p.txt" containing the values of the momentum in the chain for a sample
        - "data/vec_tr_B_g.txt" containing the values of the monitor function in the chain for a sample
        - "data/vec_tr_B_nA.txt" containing the values of the the average number of iterations for 
                                 the fixed stepsize integration A  
        - "data/vec_tr_B_dA.txt" containing the values of the average of the difference between the last
                                 values of the position in the last two iterations. 
    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory
    i: int 
        index in the vector of stepsize, required to save the vector of the position and momentum

    Return
    ------
    0: int
    */
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;

    // empty vector to save the values 
    vector<double> vec_q(numruns/printskip,0);
    vector<double> vec_p(numruns/printskip,0);
    vector<double> vec_g(numruns/printskip,0);
    vector<double> vec_nA(numruns/printskip,0);
    vector<double> vec_dA(numruns/printskip,0);

    // Initialise snapshot
    nsp=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,g) shared(ns,vec_q,vec_p,vec_g,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
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
            vector<double> resA =getstepA(q,p,dt);
            q=resA[0];
            double nA=resA[1];
            double diffA=resA[2];

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
            resA =getstepA(q,p,dt);
            q=resA[0];
            nA+=resA[1]/2;
            diffA+=resA[2]/2;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            g = getg(q);
            gp=getgprime(q);
            gdt = dt*g;
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

           //***************
            //* Save values 
            //**************
            if (nt%printskip==0){
                vec_q[nsp]=q;
                vec_p[nsp]=p;
                vec_g[nsp]=g;
                vec_nA[nsp]=nA;
                vec_dA[nsp]=diffA;
                nsp+=1;
            }
        }
    
    }


    // save the some of the values generated. 
    string path=PATH;
    fstream file;
    file << fixed << setprecision(16) << endl;

    string file_name=path+"/vec_tr_B_q.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_q.begin(), vec_q.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_B_p.txt";
    file.open(file_name,ios_base::out);
    copy(vec_p.begin(), vec_p.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_B_g.txt";
    file.open(file_name,ios_base::out);
    copy(vec_g.begin(), vec_g.end(), out_itr);
    file.close();
    
    file_name=path+"/vec_tr_B_nA.txt";
    file.open(file_name,ios_base::out);
    copy(vec_nA.begin(), vec_nA.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_B_dA.txt";
    file.open(file_name,ios_base::out);
    copy(vec_dA.begin(), vec_dA.end(), out_itr);
    file.close();

    return 0;
    }

////////////////////////////////////////////////////////
////////// ADAPTIVE WITH ADAPTIVE STEP IN O ////////////
////////////////////////////////////////////////////////

//
// \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B} 
// or BAOAB for the IP-transformed system with correction step in O
//

double num_int_tilde_baoab(double dt, double numruns)
{    /*
    This function implements the numerical integrator \tilde{B}\tilde{A}\tilde{O}\tilde{A}\tilde{B}, and save the 
    values generated under: 
        - "data/vec_noada_O_q.txt" containing the values of the position in the chain for a sample
        - "data/vec_noada_O_p.txt" containing the values of the momentum in the chain for a sample
        - "data/vec_tr_O_g.txt" containing the values of the monitor function in the chain for a sample
        - "data/vec_tr_O_nA.txt" containing the values of the the average number of iterations for 
                                 the fixed stepsize integration A  
        - "data/vec_tr_O_dA.txt" containing the values of the average of the difference between the last
                                 values of the position in the last two iterations. 
    Input
    -----
    dt: double
        value of the stepsize
    numruns: double 
        number of runs for one trajectory

    Return
    ------
    0: int
    */

    
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;


    // Savethe values 
    vector<double> vec_q(numruns/printskip,0);
    vector<double> vec_p(numruns/printskip,0);
    vector<double> vec_g(numruns/printskip,0);
    vector<double> vec_nA(numruns/printskip,0);
    vector<double> vec_dA(numruns/printskip,0);


    nsp=0; 
    #pragma omp parallel private(q,p,f,C,nt,gdt,g) shared(ns,vec_q,vec_p,vec_g,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
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
            vector<double> resA =getstepA(q,p,dt);
            q=resA[0];
            double nA=resA[1];
            double diffA=resA[2];


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
            resA =getstepA(q,p,dt);
            q=resA[0];
            nA+=resA[1]/2;
            diffA+=resA[2]/2;


            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            g = getg(q);
            gdt = dt*g;
            p += 0.5*gdt*f;

            if (nt%printskip==0){
            vec_q[nsp]=q;
            vec_p[nsp]=p;
            vec_g[nsp]=g;
            vec_nA[nsp]=nA;
            vec_dA[nsp]=diffA;
            nsp+=1;
            }
        }

    }



    // save the some of the values generated. 
    string path=PATH;
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/vec_tr_O_q.txt";

    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_q.begin(), vec_q.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_O_p.txt";
    file.open(file_name,ios_base::out);
    copy(vec_p.begin(), vec_p.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_O_g.txt";
    file.open(file_name,ios_base::out);
    copy(vec_g.begin(), vec_g.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_O_nA.txt";
    file.open(file_name,ios_base::out);
    copy(vec_nA.begin(), vec_nA.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_O_dA.txt";
    file.open(file_name,ios_base::out);
    copy(vec_dA.begin(), vec_dA.end(), out_itr);
    file.close();
    return 0;
    }



int main(void) {    
        double dti = h;
        double ni = Nt;
        // Run the three numerical integrators
        double re=num_int_tilde_baoab(dti,ni);
        re=num_int_hat_baoab(dti,ni);
        re=num_int_baoab(dti,ni);
return 0;
}

