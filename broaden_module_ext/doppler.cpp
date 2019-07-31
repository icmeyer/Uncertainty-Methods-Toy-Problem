//Doppler broadening module
// Based on "doppler.F90" from OpenMC
#include <iostream>
#include <istream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xcsv.hpp"

// Function declarations
xt::xarray<double> slbw(xt::xarray<double> energy);
xt::xarray<double> broaden(xt::xarray<double> energy,
             xt::xarray<double> xs,
             int A_target,
             float T);
void calculate_F(xt::xarray<double> &F, float a);
void broaden_c(int n, float broadened[], float original[], float temp, int A_target);

// Constants
float PI = 3.14159265359;
float SQRT_PI_INV = 1 / std::sqrt(PI);
float K_BOLTZMANN = 8.6173303e-5; // Boltzmann constant in eV/K


// int main(){
//   int n = 1000;
//   xt::xarray<double> energy = xt::linspace<double>(6.0, 7.0, n);
// 
//   xt::xarray<double> slbw_vals = slbw(energy);
// 
//   // Write cross section to file
//   std::ofstream out_file;
//   auto results = xt::stack(xtuple(energy, slbw_vals),1);
//   out_file.open("my_func_results");
//   xt::dump_csv(out_file, results);
//   out_file.close();
// 
//   //Run SIGMA1 Test
//   int A=238; float T;
//   xt::xarray<double> xsNew = broaden(energy, slbw_vals, A, T=297);
//   printf("Done broadening");
// 
//   auto results_new = xt::stack(xtuple(energy, xsNew),1);
//   out_file.open("sigma1_results");
//   xt::dump_csv(out_file, results_new);
//   out_file.close();
// 
//   printf("Done writing\n");
// }
extern "C" void broaden_c(int n, double broadened[], double energy[],
                          double original[], double temp, int A_target){
  std::size_t size = n;
  std::vector<std::size_t> shape = {size};
  // xt::xarray<double> xs = xt::xadapt(original, size, xt::acquire_ownership(), shape);
  auto xs = xt::xadapt(original, size, xt::no_ownership(), shape);
  auto energy_c = xt::xadapt(energy, size, xt::no_ownership(), shape);
  auto xs_broadened = broaden(energy_c, xs, A_target, temp);
  for(int i=0; i<n; i++){
    broadened[i] = xs_broadened[i];
  }
}

xt::xarray<double> slbw(xt::xarray<double> energy){
  //Constants
  float PI = 3.14159265359;
  float res_E = 6.673491;
  float J = 0.5;
  float ap =  0.948;
  int A = 238;
  float gn = 1.475792e-3;
  float gg = 2.3e-2;
  float gfa = 0;
  float gfb = 0;

  //Derived constants
  float sigma_pot = 4*PI*std::pow(ap,2);
  float g_tot = gn + gg + gfa + gfb;
  float r = 2603911/res_E*(A+1)/A;
  float q = 2*std::pow((r*sigma_pot), 0.5);
  xt::xarray<double> x = 2*(energy - res_E)/(g_tot);
  xt::xarray<double> psi = 1/(1+xt::pow(x,2));
  xt::xarray<double> chi = x/(1+xt::pow(x,2));

  //Calculate capture only for now
  xt::xarray<double> xs= (gn*gg)/(g_tot*g_tot)*r*psi*xt::pow((res_E/energy),0.5);
  return xs;
}

xt::xarray<double> broaden(xt::xarray<double> energy,
                           xt::xarray<double> xs,
                           int A_target,
                           float T){
  //Initialize xs
  int size = xs.size();
  xt::xarray<double> sigmaNew = xt::zeros<double>({1,size});
                           
  // Find number of energies, used for 0 indexing
  int n = energy.size() - 1;

  // Determine the alpha parameter
  float alpha = A_target/(K_BOLTZMANN * T);

  // Calculate x values
  xt::xarray<double> x = xt::pow((alpha * energy),0.5);

  // F function test, remove at some point
  // xt::xarray<double> F_a = xt::zeros<double>({5});
  // calculate_F(F_a, 2);
  // std::cout << "F test "<< F_a << std::endl;


  // Loop over incoming neutron energies
  for(int i=0; i<=n; i++){
    // std::cout << "energy "<< i << "/" << n << std::endl;
    // printf("Energy %i: %f \n", i, energy[i]);

    float sigma = 0;
    float y = x[i];
    float y_sq = y*y;
    float y_inv = 1 / y;
    float y_inv_sq = y_inv / y;
    float Ak, Bk, slope;

    //Evaluate first term from x[k] - y = 0 to -4

    int k = i;
    float a = 0;
    // xt::xarray<double> F_a, F_b, H = xt::zeros<double>({1, 5});
    xt::xarray<double> F_a = xt::zeros<double>({1, 5});
    calculate_F(F_a, a);
    xt::xarray<double> F_b = xt::zeros<double>({1, 5});
    xt::xarray<double> H = xt::zeros<double>({1, 5});
    
    while(a >= -4.0 && k > 0){
      // Move to next point
      F_b = F_a;
      k -= 1;
      a = x[k] - y;
      
      calculate_F(F_a, a);
      H = F_a - F_b;

      // Calculate A(k), B(k), and slope terms
      Ak = y_inv_sq*H[2] + 2.0*y_inv*H[1] + H[0];
      Bk = y_inv_sq*H[4] + 4.0*y_inv*H[3] + 6.0*H[2] + 4.0*y*H[1] + y_sq*H[0];
      slope  = (xs[k+1] - xs[k]) / (std::pow(x[k+1],2)-std::pow(x[k],2));
      
      //Add contribution to broadened cross section
      sigma += Ak*(xs[k] - slope*std::pow(x[k],2)) + slope*Bk;
      // printf("sigma in left moving: %f \n", sigma);
      // std::cout << "At index " << i << " sigma 1 " << sigma << std::endl;
    } // end while

    // Extend cross section to 0 assuming 1/V shape
    if (k == 0 && a >= -4.0){
      // x=0 implies that a = -y
      F_b = F_a;
      a = -y;

      // Calculate F and H functions
      calculate_F(F_a, a);
      H = F_a - F_b;

      //Add contribution to broadened cross section
      sigma += xs[k]*x[k]*(y_inv_sq*H[1] + y_inv*H[0]);
      //printf("sigma on left boundary: %f \n", sigma);
    } // end if 
    
    // Evaluate first term from x[k] - y = 0 to 4
    k = i;
    float b = 0;
    calculate_F(F_b, b);

    while(b <= 4.0 and k < n){
      // Move to next point
      F_a = F_b;
      k += 1;
      b = x[k] - y;

      // Calculate F and H functions
      calculate_F(F_b, b);
      H = F_a - F_b;

      // Calculate A(k), B(k), and slope terms
      Ak = y_inv_sq*H[2] + 2.0*y_inv*H[1] + H[0];
      Bk = y_inv_sq*H[4] + 4.0*y_inv*H[3] + 6.0*H[2] + 4.0*y*H[1] + y_sq*H[0];
      slope  = (xs[k] - xs[k-1]) / (std::pow(x[k],2)-std::pow(x[k-1],2));

      //Add contribution to broadened cross section
      sigma += Ak*(xs[k] - slope*std::pow(x[k],2)) + slope*Bk;
      //printf("sigma in right moving: %f \n", sigma);
    } // end while

    // Extend cross section to infinity assuming constant shape
    if (k == n && b <= 4.0){
      // Calculate F function at last energy point
      float a = x[k] - y;
      calculate_F(F_a, a);


      // Add contribution to broadened cross section
      // std::cout << "Test 1" << std::endl;
      // std::cout << "F_a " << F_a << " F_b " << F_b << std::endl;
      // std::cout << "xs[k] " << xs[k] << std::endl;
      sigma += xs[k]*(y_inv_sq*F_a[2] + 2.0*y_inv*F_a[1] + F_a[0]);
      //printf("sigma on right boundary: %f \n", sigma);
    } // end if

    // Evaluate second term from x[k] + y = 0 to +4
    if(y <= 4.0){
      // Swap sign on y
      y = -y;
      y_inv = -y_inv;
      k = 0;

      // Calculate a and b based on 0 and x[1]
      a = -y;
      b = x[k] - y;

      // Calculate F and H functions
      calculate_F(F_a, a);
      calculate_F(F_b, b);
      H = F_a - F_b;

      // Add contribution to broadend cross section
      sigma = sigma - xs[k]*xs[k]*(y_inv_sq*H[1] + y_inv*H[0]);

      // Now progress forward doing the remainder of the second term
      while(b <= 4.0){
        // Move to next point
        F_a = F_b;
        k += 1;
        b = x[k] - y;
        // Calculate F and H functions
        calculate_F(F_b, b);
        H = F_a - F_b;

        // Calculate A[k], B[k], and slope terms
        Ak = y_inv_sq*H[2] + 2.0*y_inv*H[1] + H[0];
        Bk = y_inv_sq*H[4] + 4.0*y_inv*H[3] + 6.0*H[2] + 4.0*y*H[1] + y_sq*H[0];
        slope  = (xs[k] - xs[k-1]) / (std::pow(x[k],2)-std::pow(x[k-1],2));

        //Add contribution to broadened cross section
        sigma = sigma - Ak*(xs[k] - slope*std::pow(x[k],2)) - slope*Bk;
      } // end while
    } // end if 
      
    // Set broadened cross section
    sigmaNew[i] = sigma;

  } // end for loop over energies
  return sigmaNew;
} // end broaden()

void calculate_F(xt::xarray<double> &F, float a){
  F[0] = 0.5*std::erfc(a);
  F[1] = 0.5*SQRT_PI_INV*std::exp(-a*a);
  F[2] = 0.5*F[0] + a*F[1];
  F[3] = F[1]*(1.0 + a*a);
  F[4] = 0.75*F[0] + F[1]*a*(1.5 + a*a);
}

// Some helpful print statements for debug
//printf("F_a : %f %f %f %f %f \n", F_a[0], F_a[1], F_a[2], F_a[3], F_a[4]);
//printf("a: %f \n", a);
//printf("y: %f \n", y);
//printf("y_inv_sq: %f \n", y_inv_sq);
//printf("x[k]: %f \n", x[k]);
//printf("sigma on right boundary before: %f \n", sigma);
