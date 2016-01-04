/** CPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_cpu.c
 *
 * \author Kyle E. Niemeyer
 * \date 09/24/2012
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity, using the finite difference method.
 * 
 * To change the grid resolution, modify "NUM". In addition, the problem is controlled
 * by the Reynolds number ("Re_num").
 * 
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, Philadelphia,
 * PA, 1998.
 * 
 * Boundary conditions:
 * u = 0 and v = 0 at x = 0, x = L, y = 0
 * u = 1.0 at y = H
 * v = 0 at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <time.h>

 #include "timer.h"

/** Problem size along one side; total number of cells is this squared */
#define NUM 512

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
	#define Real double

 	#define ZERO 0.0
	#define ONE 1.0
	#define TWO 2.0
 	#define FOUR 4.0

 	#define SMALL 1.0e-10

	/** Reynolds number */
	#define Re_num 1000.0

	/** SOR relaxation parameter */
	#define omega 1.7

	/** Discretization mixture parameter (gamma) */
	#define mix_param 0.9

	/** Safety factor for time step modification */
	#define tau 0.5

	/** Body forces in x- and y- directions */
	#define gx 0.0
	#define gy 0.0

	/** Domain size (non-dimensional) */
	#define xLength 1.0
	#define yLength 1.0
#else
	#define Real float

	// replace double functions with float versions
	#undef fmin
	#define fmin fminf
	#undef fmax
	#define fmax fmaxf
	#undef fabs
	#define fabs fabsf
	#undef sqrt
	#define sqrt sqrtf

 	#define ZERO 0.0f
	#define ONE 1.0f
	#define TWO 2.0f
	#define FOUR 4.0f
	#define SMALL 1.0e-10f;

		/** Reynolds number */
	#define Re_num 1000.0f

	/** SOR relaxation parameter */
	#define omega 1.7f

	/** Discretization mixture parameter (gamma) */
	#define mix_param 0.9f

	/** Safety factor for time step modification */
	#define tau 0.5f

	/** Body forces in x- and y- directions */
	#define gx 0.0f
	#define gy 0.0f

	/** Domain size (non-dimensional) */
	#define xLength 1.0f
	#define yLength 1.0f
#endif

// OpenACC
#ifdef _OPENACC
	#include <openacc.h>
#endif

// OpenMP
#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_num_threads() 1
#endif

#if __STDC_VERSION__ < 199901L
	#define restrict __restrict__
#endif

/** Max macro (type safe, from GNU) */
//#define MAX(a,b)  __extension__({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
//#define MIN(a,b)  __extension__({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

/** Mesh sizes */
#define dx (xLength / NUM)
#define dy (yLength / NUM)

#define SIZE ((NUM * NUM) + (4 * NUM) + 4)
#define SIZEP ((NUM * NUM / 2) + (3 * NUM) + 4)

// map two-dimensional indices to one-dimensional memory
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

///////////////////////////////////////////////////////////////////////////////
void set_BCs (Real* restrict u, Real* restrict v)
{
	int row, col;

	// loop through rows
	#pragma omp parallel for shared(u, v) private(row)
	//#pragma acc kernels present(u[0:SIZE], v[0:SIZE])
	#pragma acc parallel present(u[0:SIZE], v[0:SIZE]) private(row, col)
	#ifdef _OPENACC
	{
	#endif
	#pragma acc loop
	for (row = 0; row < NUM + 2; ++row) {

		// left boundary
		u(0, row) = ZERO;
		v(0, row) = -v(1, row);

		// right boundary
		u(NUM, row) = ZERO;
		v(NUM + 1, row) = -v(NUM, row);

	} // end for row

	#pragma omp parallel for shared(u, v) private(col)
	//#pragma acc kernels present(u[0:SIZE], v[0:SIZE])
	#pragma acc loop
	for (col = 0; col < NUM + 2; ++col) {

		// bottom boundary
		u(col, 0) = -u(col, 1);
		v(col, 0) = ZERO;

		// top boundary
		u(col, NUM + 1) = TWO - u(col, NUM);
		v(col, NUM) = ZERO;

	} // end for col
	#ifdef _OPENACC
	}
	#endif

} // end set_BCs

///////////////////////////////////////////////////////////////////////////////

void calculate_F (const Real dt, const Real* restrict u, const Real* restrict v, 
				  Real* restrict F)
{	
	int row, col;

	#pragma omp parallel for shared(dt, u, v, F) \
			private(row, col)
	#pragma acc kernels present(u[0:SIZE], v[0:SIZE], F[0:SIZE])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < NUM + 1; ++row) {

			// right boundary
			if (col == NUM) {
				// also do left boundary (col = 0)
				F(0, row) = u(0, row);
				F(NUM, row) = u(NUM, row);

			} else {
				
				// u velocities
				Real u_ij = u(col, row);
				Real u_ip1j = u(col + 1, row);
				Real u_ijp1 = u(col, row + 1);
				Real u_im1j = u(col - 1, row);
				Real u_ijm1 = u(col, row - 1);

				// v velocities
				Real v_ij = v(col, row);
				Real v_ip1j = v(col + 1, row);
				Real v_ijm1 = v(col, row - 1);
				Real v_ip1jm1 = v(col + 1, row - 1);
				
				// finite differences
				Real du2dx, duvdy, d2udx2, d2udy2;

				du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
						+ mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
						- fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (FOUR * dx);
				duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
						+ mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
						- fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (FOUR * dy);
				d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (dx * dx);
				d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (dy * dy);

				F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
			} // end if
		} // end for col
	} // end for row
		
} // end calculate_F

///////////////////////////////////////////////////////////////////////////////

void calculate_G (const Real dt, const Real* restrict u, const Real* restrict v, 
				  Real* restrict G)
{
	int col, row;

	#pragma omp parallel for shared(dt, u, v, G) \
			private(col, row)
	#pragma acc kernels present(u[0:SIZE], v[0:SIZE], G[0:SIZE])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < NUM + 1; ++row) {

			// upper boundary
			if (row == NUM) {
				// also do bottom boundary
				G(col, 0) = v(col, 0);
				G(col, NUM) = v(col, NUM);

			} else {
				
				// u velocities
				Real u_ij = u(col, row);
				Real u_ijp1 = u(col, row + 1);
				Real u_im1j = u(col - 1, row);
				Real u_im1jp1 = u(col - 1, row + 1);

				// v velocities
				Real v_ij = v(col, row);
				Real v_ijp1 = v(col, row + 1);
				Real v_ip1j = v(col + 1, row);
				Real v_ijm1 = v(col, row - 1);
				Real v_im1j = v(col - 1, row);
				
				// finite differences
				Real dv2dy, duvdx, d2vdx2, d2vdy2;
			
				dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
						+ mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
						- fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (FOUR * dy);
				duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
						+ mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
						- fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (FOUR * dx);
				d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
				d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);

				G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
			} // end if
		} // end for row
	} // end for col
		
} // end calculate_G

///////////////////////////////////////////////////////////////////////////////

Real sum_pressure (const Real* restrict pres_red, const Real* restrict pres_black)
{
	int row, col;
	Real sum = ZERO;

	int NUM_2 = NUM >> 1;

	#pragma omp parallel for shared(pres_black, pres_red, sum) \
			reduction(+:sum) private(col, row)
	#pragma acc kernels present(pres_red[0:SIZEP], pres_black[0:SIZEP])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < (NUM / 2) + 1; ++row) {
			
			Real pres_r = pres_red(col, row);
			Real pres_b = pres_black(col, row);

			sum += (pres_r * pres_r) + (pres_b * pres_b);
		} // end for row
	} // end for col

	return sum;
} // end sum_pressure

///////////////////////////////////////////////////////////////////////////////

void set_pressure_BCs (Real* restrict pres_red, Real* restrict pres_black)
{
	int row, col;

	int NUM_2 = NUM >> 1;

	// loop over columns
	#pragma omp parallel for shared(pres_black, pres_red) private(col)
	#pragma acc parallel present(pres_black[0:SIZEP], pres_red[0:SIZEP]) private(row, col)
	#ifdef _OPENACC
	{
	#endif
	#pragma acc loop
	for (col = 1; col < NUM + 1; col += 2) {

		// p_i,0 = p_i,1
		pres_black(col, 0) = pres_red(col, 1);
		pres_red(col + 1, 0) = pres_black(col + 1, 1);

		// p_i,jmax+1 = p_i,jmax
		pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
		pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);

	} // end for col

	// loop over rows
	#pragma omp parallel for shared(pres_black, pres_red) private(row)
	//#pragma acc kernels present(pres_black[0:SIZEP], pres_red[0:SIZEP])
	#pragma acc loop
	for (row = 1; row < (NUM / 2) + 1; ++row) {

		// p_0,j = p_1,j
		pres_black(0, row) = pres_red(1, row);
		pres_red(0, row) = pres_black(1, row);

		// p_imax+1,j = p_imax,j
		pres_black((NUM + 1), row) = pres_red(NUM, row);
		pres_red((NUM + 1), row) = pres_black(NUM, row);

	} // end for row
	#ifdef _OPENACC
	}
	#endif
 
} // end set_pressure_BCs

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for red cells
 * 
 * \param[in]			dt					time-step size
 * \param[in]			F						array of discretized x-momentum eqn terms
 * \param[in]			G						array of discretized y-momentum eqn terms
 * \param[in]			pres_black	pressure values of black cells
 * \param[inout]	pres_red		pressure values of red cells
 * \param[inout]	norm_L2			variable holding summed residuals
 */
void red_kernel	(const Real dt, const Real* restrict F, 
				 const Real* restrict G, const Real* restrict pres_black,
				 Real* restrict pres_red)
{
	int col, row;

	int NUM_2 = NUM >> 1;
	
	// loop over actual cells, skip boundary cells
	#pragma omp parallel for shared(dt, F, G, pres_black, pres_red) \
			private(col, row)
	#pragma acc kernels present(F[0:SIZE], G[0:SIZE], pres_black[0:SIZEP], pres_red[0:SIZEP])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < (NUM / 2) + 1; ++row) {
		
			Real p_ij = pres_red(col, row);
			
			Real p_im1j = pres_black(col - 1, row);
			Real p_ip1j = pres_black(col + 1, row);
			Real p_ijm1 = pres_black(col, row - (col & 1));
			Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));
			
			// right-hand side
			Real rhs = (((F(col, (2 * row) - (col & 1))
					    - F(col - 1, (2 * row) - (col & 1))) / dx)
					  + ((G(col, (2 * row) - (col & 1))
					    - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;
			
			pres_red(col, row) = p_ij * (ONE - omega) + omega * 
				(((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
				rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));

		} // end for row
	} // end for col

} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for black cells
 * 
 * \param[in]			dt					time-step size
 * \param[in]			F						array of discretized x-momentum eqn terms
 * \param[in]			G						array of discretized y-momentum eqn terms
 * \param[in]			pres_red		pressure values of red cells
 * \param[inout]	pres_black	pressure values of black cells
 * \param[inout]	norm_L2			variable holding summed residuals
 */
void black_kernel (const Real dt, const Real* restrict F, 
				   const Real* restrict G, const Real* restrict pres_red, 
				   Real* restrict pres_black)
{
	int col, row;

	int NUM_2 = NUM >> 1;
	
	// loop over actual cells, skip boundary cells
	#pragma omp parallel for shared(dt, F, G, pres_red, pres_black) \
			private(col, row)
	#pragma acc kernels present(F[0:SIZE], G[0:SIZE], pres_red[0:SIZEP], pres_black[0:SIZEP])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < (NUM / 2) + 1; ++row) {
			
			Real p_ij = pres_black(col, row);

			Real p_im1j = pres_red(col - 1, row);
			Real p_ip1j = pres_red(col + 1, row);
			Real p_ijm1 = pres_red(col, row - ((col + 1) & 1));
			Real p_ijp1 = pres_red(col, row + (col & 1));
			
			// right-hand side
			Real rhs = (((F(col, (2 * row) - ((col + 1) & 1))
					 	- F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
					  + ((G(col, (2 * row) - ((col + 1) & 1))
					    - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;
	
			pres_black(col, row) = p_ij * (ONE - omega) + omega * 
								   (((p_ip1j + p_im1j) / (dx * dx))
								 + ((p_ijp1 + p_ijm1) / (dy * dy)) - rhs)
								 / ((TWO / (dx * dx)) + (TWO / (dy * dy)));
			
		} // end for row
	} // end for col
	
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

void SOR (const Real dt, const Real* restrict F, const Real* restrict G,
			Real* restrict pres_red, Real* restrict pres_black)
{
	Real norm_L2 = ZERO;
	int col, row;
	
	int NUM_2 = NUM >> 1;

	// loop over actual cells, skip boundary cells
	for (col = 1; col < NUM + 1; ++col) {
		for (row = 1; row < NUM + 1; ++row) {

			Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs;

			if ((row + col) & 1 == 0) {
				p_ij = pres_red(col, (row + (col & 1)) >> 1);

				p_im1j = pres_black(col - 1, (row + (col & 1)) >> 1);
				p_ip1j = pres_black(col + 1, (row + (col & 1)) >> 1);
				p_ijm1 = pres_black(col, (row - 1 + ((col + 1) & 1)) >> 1);
				p_ijp1 = pres_black(col, (row + 1 + ((col + 1) & 1)) >> 1);
				
				// right-hand side
				rhs = (((F(col, row) - F(col - 1, row)) / dx)
					+ ((G(col, row) - G(col, row - 1)) / dy)) / dt;
				
				p_ij = p_ij * (ONE - omega) + omega * 
					(((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
					rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));

				pres_red(col, (row + (col & 1)) >> 1) = p_ij;
			} else {
				p_ij = pres_black(col, (row + ((col + 1) & 1)) >> 1);

				p_im1j = pres_red(col - 1, (row + ((col + 1) & 1)) >> 1);
				p_ip1j = pres_red(col + 1, (row + ((col + 1) & 1)) >> 1);
				p_ijm1 = pres_red(col, (row - 1 + (col & 1)) >> 1);
				p_ijp1 = pres_red(col, (row + 1 + (col & 1)) >> 1);
				
				// right-hand side
				rhs = (((F(col, row) - F(col - 1, row)) / dx)
					+ ((G(col, row) - G(col, row - 1)) / dy)) / dt;
				

				p_ij = p_ij * (ONE - omega) + omega * 
					(((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
					rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));

				pres_black(col, (row + ((col + 1) & 1)) >> 1) = p_ij;
			}

		} // end for row
	} // end for col

}

///////////////////////////////////////////////////////////////////////////////

Real calc_residual (const Real dt, const Real* restrict F, const Real* restrict G, 
					const Real* restrict pres_red, const Real* restrict pres_black)
{	
	int row, col;
	Real residual = ZERO;

	int NUM_2 = NUM >> 1;

	#pragma omp parallel for shared(dt, F, G, pres_red, pres_black, residual) \
			reduction(+:residual) private(col, row)
	#pragma acc kernels present(F[0:SIZE], G[0:SIZE], pres_red[0:SIZEP], pres_black[0:SIZEP])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < (NUM / 2) + 1; ++row) {

			Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res, res2;

			// red point
			p_ij = pres_red(col, row);

			p_im1j = pres_black(col - 1, row);
			p_ip1j = pres_black(col + 1, row);
			p_ijm1 = pres_black(col, row - (col & 1));
			p_ijp1 = pres_black(col, row + ((col + 1) & 1));

			rhs = (((F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) / dx)
				+  ((G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

			// calculate residual
			res = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
				+ ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

			// black point
			p_ij = pres_black(col, row);

			p_im1j = pres_red(col - 1, row);
			p_ip1j = pres_red(col + 1, row);
			p_ijm1 = pres_red(col, row - ((col + 1) & 1));
			p_ijp1 = pres_red(col, row + (col & 1));	

			// right-hand side
			rhs = (((F(col, (2 * row) - ((col + 1) & 1)) - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
				+  ((G(col, (2 * row) - ((col + 1) & 1)) - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

			// calculate residual
			res2 = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
				 + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

			residual += (res * res) + (res2 * res2);

		} // end for row
	} // end for col

	return residual;
}

///////////////////////////////////////////////////////////////////////////////

Real calculate_u (const Real dt, const Real* restrict F, 
				  const Real* restrict pres_red, const Real* restrict pres_black, 
				  Real* restrict u)
{
	Real max_u = SMALL;
	int col, row;

	int NUM_2 = NUM >> 1;
	
	// loop over actual cells, skip boundary cells
	#pragma omp parallel for shared(dt, F, pres_black, pres_red, u, max_u) \
			private(col, row)
	#pragma acc kernels present(F[0:SIZE], pres_red[0:SIZEP], pres_black[0:SIZEP], u[0:SIZE])
	#pragma acc loop independent
	//for (col = 1; col < NUM + 1 ; ++col) {
	for (col = 1; col < NUM; ++col) {
		#pragma acc loop independent
		for (row = 1; row < (NUM / 2) + 1; ++row) {

			Real p_ij, p_ip1j, new_u, new_u2;

			// red point
			p_ij = pres_red(col, row);
			p_ip1j = pres_black(col + 1, row);

			new_u = F(col, (2 * row) - (col & 1)) - (dt * (p_ip1j - p_ij) / dx);
			u(col, (2 * row) - (col & 1)) = new_u;

			// black point
			p_ij = pres_black(col, row);
			p_ip1j = pres_red(col + 1, row);

			new_u2 = F(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ip1j - p_ij) / dx);
			u(col, (2 * row) - ((col + 1) & 1)) = new_u2;

			// check for max of these two
			new_u = fmax(fabs(new_u), fabs(new_u2));

			if ((2 * row) == NUM) {
				// also test for max velocity at vertical boundary
				new_u = fmax(new_u, fabs( u(col, NUM + 1) ));
			}

			// get maximum u velocity
			max_u = fmax(max_u, new_u);

		} // end for row
	} // end for col

	#pragma acc kernels present(u[0:SIZE])
	#pragma acc loop independent
	for (row = 1; row < (NUM / 2) + 1; ++row) {
		col = NUM;

		// check for maximum velocity in boundary cells also
		Real test_u = fmax(fabs( u(NUM, (2 * row)) ), fabs( u(0, (2 * row)) ));
		test_u = fmax(fabs( u(NUM, (2 * row) - 1) ), test_u);
		test_u = fmax(fabs( u(0, (2 * row) - 1) ), test_u);

		test_u = fmax(fabs( u(NUM + 1, (2 * row)) ), test_u);
		test_u = fmax(fabs( u(NUM + 1, (2 * row) - 1) ), test_u);

		max_u = fmax(max_u, test_u);
	} // end for row



	
	return max_u;
} // end calculate_u

///////////////////////////////////////////////////////////////////////////////

Real calculate_v (const Real dt, const Real* restrict G, 
				  const Real* restrict pres_red, const Real* restrict pres_black, 
				  Real* restrict v)
{
	Real max_v = SMALL;
	int col, row;

	int NUM_2 = NUM >> 1;
	
	// loop over actual cells, skip boundary cells
	#pragma omp parallel for shared(dt, G, pres_black, pres_red, v, max_v, NUM_2) \
			private(col, row)
	#pragma acc kernels present(G[0:SIZE], pres_red[0:SIZEP], pres_black[0:SIZEP], v[0:SIZE])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		//for (row = 1; row < (NUM / 2) + 1; ++row) {
		for (row = 1; row < (NUM / 2); ++row) {

			Real p_ij, p_ijp1, new_v, new_v2;

			// red pressure point
			p_ij = pres_red(col, row);
			p_ijp1 = pres_black(col, row + ((col + 1) & 1));
		
			new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
			v(col, (2 * row) - (col & 1)) = new_v;


			// black pressure point
			p_ij = pres_black(col, row);
			p_ijp1 = pres_red(col, row + (col & 1));
			
			new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
			v(col, (2 * row) - ((col + 1) & 1)) = new_v2;


			// check for max of these two
			new_v = fmax(fabs(new_v), fabs(new_v2));

			if (col == NUM) {
				// also test for max velocity at vertical boundary
				new_v = fmax(new_v, fabs( v(NUM + 1, (2 * row)) ));
			}

			// get maximum v velocity
			max_v = fmax(max_v, new_v);

		} // end for row
	} // end for col

	row = NUM_2;

	#pragma omp parallel for shared(dt, G, pres_black, pres_red, v, max_v, row, NUM_2) \
			private(col)
	#pragma acc kernels present(G[0:SIZE], pres_red[0:SIZEP], pres_black[0:SIZEP], v[0:SIZE])
	#pragma acc loop independent
	for (col = 1; col < NUM + 1; ++col) {

		Real new_v;

		if ((col % 2) == 1) {
			// black point is on boundary, only calculate red point below it
			Real p_ij = pres_red(col, row);
			Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));
		
			new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
			v(col, (2 * row) - (col & 1)) = new_v;
		} else {
			// red point is on boundary, only calculate black point below it
			Real p_ij = pres_black(col, row);
			Real p_ijp1 = pres_red(col, row + (col & 1));
		
			new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
			v(col, (2 * row) - ((col + 1) & 1)) = new_v;
		}

		// get maximum v velocity
		Real test_v = fabs(new_v);

		// check for maximum velocity in boundary cells also
		test_v = fmax(fabs( v(col, NUM) ), test_v);
		test_v = fmax(fabs( v(col, 0) ), test_v);

		test_v = fmax(fabs( v(col, NUM + 1) ), test_v);

		max_v = fmax(max_v, test_v);
		
	} // end for col
	
	return max_v;
} // end calculate_v

///////////////////////////////////////////////////////////////////////////////

int main (int argc, char *argv[])
{
	
	// iterations for Red-Black Gauss-Seidel with SOR
	int iter = 0;
	const int it_max = 100000;
	
	// SOR iteration tolerance
	const Real tol = 0.001;
	
	// time range
	const Real time_start = 0.0;
	const Real time_end = 20.0;
	
	// initial time step size
	Real dt = 0.02;
	
	int size = (NUM + 2) * (NUM + 2);
	int size_pres = ((NUM / 2) + 2) * (NUM + 2);
	
	// arrays for pressure and velocity
	Real* restrict F;
	Real* restrict u;
	Real* restrict G;
	Real* restrict v;
	
	F = (Real *) calloc (size, sizeof(Real));
	u = (Real *) calloc (size, sizeof(Real));
	G = (Real *) calloc (size, sizeof(Real));
	v = (Real *) calloc (size, sizeof(Real));
	
	for (int i = 0; i < size; ++i) {
		F[i] = ZERO;
		u[i] = ZERO;
		G[i] = ZERO;
		v[i] = ZERO;
	}
	
	// arrays for pressure
	Real* restrict pres_red;
	Real* restrict pres_black;
	
	pres_red = (Real *) calloc (size_pres, sizeof(Real));
	pres_black = (Real *) calloc (size_pres, sizeof(Real));
	
	for (int i = 0; i < size_pres; ++i) {
		pres_red[i] = ZERO;
		pres_black[i] = ZERO;
	}

	// set and initialize GPU, using device based on command line argument
	#ifdef _OPENACC
		acc_init (acc_device_nvidia);
		int id = 0;

		if (argc > 1) {
			int num_devices = acc_get_num_devices (acc_device_nvidia);
			id = *(argv[1]) - '0';

			if ((id <= 0) || (id >= num_devices)) {
				// not in range
				printf("Error: GPU device number not in correct range\n");
				printf("Provide number between 0 and %i\n", num_devices - 1);
				exit(1);
			}
		}

		acc_set_device_num (id, acc_device_nvidia);
	#endif

	// set number of threads for OpenMP based on command line argument
	// if no argument, use the environment default (OMP_NUM_THREADS)
	#ifdef _OPENMP
		int max_threads = omp_get_max_threads ();
		int num_threads = max_threads;

		if (argc > 1) {
			// first check if is number
			num_threads = *(argv[1]) - '0';

			if ((num_threads <= 0) || (num_threads > max_threads)) {
				// not a digit, error
				printf("Error: Number of threads not in correct range\n");
				printf("Provide number between 1 and %i\n", max_threads);
				exit(1);
			}
		}

		omp_set_num_threads (num_threads);
	#endif	


	// print problem info
	printf("Problem size: %d x %d \n", NUM, NUM);
	printf("Num threads: %d\n", omp_get_num_threads());
	
	//////////////////////////////
	// start timer
	//clock_t start_time = clock();
	StartTimer();
	//////////////////////////////
	
	// time-step size based on grid and Reynolds number
	Real dt_Re = 0.5 * Re_num / ((ONE / (dx * dx)) + (ONE / (dy * dy)));
	
	Real time = time_start;

	Real max_u = SMALL;
	Real max_v = SMALL;
	
	// time iteration loop
	#pragma acc data copyin(F[0:SIZE], G[0:SIZE]) \
			copy(u[0:SIZE], v[0:SIZE], pres_red[0:SIZEP], pres_black[0:SIZEP])
	{

	// set boundary conditions
	set_BCs (u, v);

	// get max velocities
	int row, col;
	//#pragma omp parallel for shared(u) private(col, row)
	#pragma acc kernels present(u[0:SIZE])
	#pragma acc loop independent
	for (col = 0; col < NUM + 2; ++col) {
		#pragma acc loop independent
		for (row = 1; row < NUM + 2; ++row) {
			max_u = fmax(max_u, fabs( u(col, row) ));
		}
	}

	//#pragma omp parallel for shared(v) private(col, row)
	#pragma acc kernels present(v[0:SIZE])
	#pragma acc loop independent
	for (col = 1; col < NUM + 2; ++col) {
		#pragma acc loop independent
		for (row = 0; row < NUM + 2; ++row) {
			max_v = fmax(max_v, fabs( v(col, row) ));
		}
	}

	// time loop
	while (time < time_end) {

		// calculate time step
		dt = fmin((dx / max_u), (dy / max_v));
		dt = tau * fmin(dt_Re, dt);
		
		if ((time + dt) >= time_end) {
			dt = time_end - time;
		}

		// calculate F and G
		calculate_F (dt, u, v, F);
		calculate_G (dt, u, v, G);

		//#pragma acc wait
		
		/////////////////////////
		// calculate new pressure
		/////////////////////////

		// get L2 norm of initial pressure
		Real p0_norm = sum_pressure (pres_red, pres_black);
		p0_norm = sqrt(p0_norm / ((Real)(NUM * NUM)));
		if (p0_norm < 0.0001) {
		   p0_norm = 1.0;
		}

		// red-black Gauss-Seidel with SOR iteration loop
		Real norm_L2;
		for (iter = 1; iter <= it_max; ++iter) {

			norm_L2 = ZERO;

			// set pressure boundary conditions
			set_pressure_BCs (pres_red, pres_black);

			// update red cells
			red_kernel (dt, F, G, pres_black, pres_red);


			// update black cells
			black_kernel (dt, F, G, pres_red, pres_black);

			//SOR (dt, F, G, pres_red, pres_black);
			norm_L2 = calc_residual (dt, F, G, pres_red, pres_black);
			
			// calculate residual
			norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

			// if tolerance has been reached, end SOR iterations
			if (norm_L2 < tol) {
				break;
			}
		} // end for		
		
		printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

		// calculate new u and v velocities
		max_u = calculate_u (dt, F, pres_red, pres_black, u);
		max_v = calculate_v (dt, G, pres_red, pres_black, v);

		// set boundary conditions
		set_BCs (u, v);

		// increase time
		time += dt;

		// single time step
		//break;
		
	} // end while
	}
	
	/////////////////////////////////
	// end timer
	//clock_t end_time = clock();
	double runtime = GetTimer();
	/////////////////////////////////
	
	#if defined(_OPENMP)
		printf("OpenMP\n");
	#elif defined(_OPENACC)
		printf("OpenACC\n");
	#else
		printf("CPU\n");
	#endif
	
	printf("Total time: %f s\n", runtime / 1000);
	
	
	// write data to file
	FILE * pfile;
	pfile = fopen("velocity_cpu.dat", "w");
	fprintf(pfile, "#x\ty\tu\tv\n");
	if (pfile != NULL) {
		for (int row = 1; row < NUM + 1; ++row) {
			for (int col = 1; col < NUM + 1; ++col) {
				
				Real u_ij = u[(col * NUM) + row];
				Real u_im1j;
				if (col == 1) {
					u_im1j = ZERO;
				} else {
					u_im1j = u[(col - 1) * NUM + row];
				}
				
				u_ij = (u_ij + u_im1j) / TWO;
				
				Real v_ij = v[(col * NUM) + row];
				Real v_ijm1;
				if (row == 1) {
					v_ijm1 = ZERO;
				} else {
					v_ijm1 = v[(col * NUM) + row - 1];
				}
				
				v_ij = (v_ij + v_ijm1) / TWO;
				
				fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
			}
		}
	}
	
	fclose(pfile);
	
	free(pres_red);
	free(pres_black);
	free(u);
	free(v);
	free(F);
	free(G);
	
	return 0;
}
