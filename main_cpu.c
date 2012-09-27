/** CPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_cpu.c
 *
 * \author Kyle E. Niemeyer
 * \date 09/24/2012
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity, using the finite difference method.
 * 
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, Philadelphia,
 * PA, 1998.
 * 
 * Boundary conditions:
 * u = 0 and v = 0 at x = 0, x = L, y = 0
 * u = ustar at y = H
 * v = 0 at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/** Problem size along one side; total number of cells is this squared */
#define NUM 64

/** Double precision */
//#define DOUBLE

#ifdef DOUBLE
	#define Real double
#else
	#define Real float
#endif

typedef unsigned int uint;

/** Reynolds number */
const Real Re_num = 1000.0;

/** SOR relaxation parameter */
const Real omega = 1.7;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9;

/** Safety factor for time step modification */
const Real tau = 0.5;

/** Body forces in x- and y- directions */
const Real gx = 0.0;
const Real gy = 0.0;

/** Normalized lid velocity */
const Real uStar = 1.0;

/** Domain size (non-dimensional) */
#define xLength 1.0
#define yLength 1.0

/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

/** Max macro (type safe, from GNU) */
#define MAX(a,b)  __extension__({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
#define MIN(a,b)  __extension__({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

///////////////////////////////////////////////////////////////////////////////

void calculate_F (const Real * u, const Real * v, const Real dt, 
									 Real * F)
{
	for (uint row = 0; row < NUM; ++row) {
		for (uint col = 0; col < NUM - 1; ++col) {
			
			// u and v velocities
			Real u_ij = u[(col * NUM) + row];
			Real u_ip1j = u[((col + 1) * NUM) + row];
			
			Real v_ij = v[(col * NUM) + row];
			Real v_ip1j = v[((col + 1) * NUM) + row];
			
			// left boundary
			Real u_im1j;
			if (col == 0) {
				u_im1j = 0.0;
			} else {
				u_im1j = u[((col - 1) * NUM) + row];
			}
			
			// bottom boundary
			Real u_ijm1, v_ijm1, v_ip1jm1;
			if (row == 0) {
				u_ijm1 = -u_ij;
				v_ijm1 = 0.0;
				v_ip1jm1 = 0.0;
			} else {
				u_ijm1 = u[(col * NUM) + row - 1];
				v_ijm1 = v[(col * NUM) + row - 1];
				v_ip1jm1 = v[((col + 1) * NUM) + row - 1];
			}
			
			// top boundary
			Real u_ijp1;
			if (row == (NUM - 1)) {
				u_ijp1 = (2 * uStar) - u_ij;
			} else {
				u_ijp1 = u[(col * NUM) + row + 1];
			}
			
			// finite differences
			Real du2dx, duvdy, d2udx2, d2udy2;

			du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
							+ mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
							- fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (4.0 * dx);
			duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
						+ mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
						- fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (4.0 * dy);
		  d2udx2 = (u_ip1j - (2.0 * u_ij) + u_im1j) / (dx * dx);
		  d2udy2 = (u_ijp1 - (2.0 * u_ij) + u_ijm1) / (dy * dy);

			F[(col * NUM) + row] = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
			
		} // end for col
		
		F[((NUM - 1) * NUM) + row] = u[((NUM - 1) * NUM) + row];
	} // end for row
		
} // end calculate_F

///////////////////////////////////////////////////////////////////////////////

void calculate_G (const Real * u, const Real * v, const Real dt, 
									Real * G)
{
	for (uint col = 0; col < NUM; ++col) {
		for (uint row = 0; row < NUM - 1; ++row) {
			
			// u and v velocities
			Real u_ij = u[(col * NUM) + row];
			Real u_ijp1 = u[(col * NUM) + row + 1];
			
			Real v_ij = v[(col * NUM) + row];
			Real v_ijp1 = v[(col * NUM) + row + 1];
			
			// bottom boundary
			Real v_ijm1;
			if (row == 0) {
				v_ijm1 = 0.0;
			} else {
				v_ijm1 = v[(col * NUM) + row - 1];
			}
			
			// left boundary
			Real v_im1j, u_im1j, u_im1jp1;
			if (col == 0) {
				v_im1j = -v_ij;
				u_im1j = 0.0;
				u_im1jp1 = 0.0;
			} else {
				v_im1j = v[((col - 1) * NUM) + row];
				u_im1j = u[((col - 1) * NUM) + row];
				u_im1jp1 = u[((col - 1) * NUM) + row + 1];
			}
			
			// right boundary
			Real v_ip1j;
			if (col == (NUM - 1)) {
				v_ip1j = -v_ij;
			} else {
				v_ip1j = v[((col + 1) * NUM) + row];
			}
			
			// finite differences
			Real dv2dy, duvdx, d2vdx2, d2vdy2;
		
			dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
			  		+ mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
						- fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (4.0 * dy);
			duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
						+ mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
						- fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (4.0 * dx);
		  d2vdx2 = (v_ip1j - (2.0 * v_ij) + v_im1j) / (dx * dx);
		  d2vdy2 = (v_ijp1 - (2.0 * v_ij) + v_ijm1) / (dy * dy);

			G[(col * NUM) + row] = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
			
		} // end for row
		
		G[(col * NUM) + NUM - 1] = v[(col * NUM) + NUM - 1];
	} // end for col
		
} // end calculate_G

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for red cells
 * 
 * \param[in]			F						array of discretized x-momentum eqn terms
 * \param[in]			G						array of discretized y-momentum eqn terms
 * \param[in]			pres_black	pressure values of black cells
 * \param[inout]	pres_red		pressure values of red cells
 * \param[inout]	norm_L2			variable holding summed residuals
 */
void red_kernel (const Real dt, const Real * F, const Real * G, const Real * pres_black,
								 Real * pres_red, Real * norm_L2)
{
	// loop over all cells
	for (uint col = 0; col < NUM; ++col) {
		for (uint row = 0; row < (NUM / 2); ++row) {
			
			uint ind_red = (col * (NUM >> 1)) + row;  					// local (red) index
			uint ind = (col * NUM) + (2 * row) + (col & 1);		// global index
			
			Real p_ij = pres_red[ind_red];			
			
			// left boundary
			Real p_im1j;
			Real F_im1j;
			if (col == 0) {
				p_im1j = p_ij;
				F_im1j = 0.0;
			} else {
				p_im1j = pres_black[((col - 1) * (NUM >> 1)) + row];
				F_im1j = F[((col - 1) * NUM) + (2 * row) + (col & 1)];
			}
			
			// right boundary
			Real p_ip1j;
			if (col == (NUM - 1)) {
				p_ip1j = p_ij;
			} else {
				p_ip1j = pres_black[((col + 1) * (NUM >> 1)) + row];
			}
			
			// bottom boundary
			Real p_ijm1;
			Real G_ijm1;
			if (((2 * row) + (col & 1)) == 0) {
				p_ijm1 = p_ij;
				G_ijm1 = 0.0;
			} else {
				p_ijm1 = pres_black[(col * (NUM >> 1)) + row - ((col + 1) & 1)];
				G_ijm1 = G[(col * NUM) + (2 * row) + (col & 1) - 1];
			}
			
			// top boundary
			Real p_ijp1;
			if (((2 * row) + (col & 1)) == (NUM - 1)) {
				p_ijp1 = p_ij;
			} else {
				p_ijp1 = pres_black[(col * (NUM >> 1)) + row + (col & 1)];
			}
			
			// right-hand side
			Real rhs = (((F[ind] - F_im1j) / dx) + ((G[ind] - G_ijm1) / dy)) / dt;
			
			pres_red[ind_red] = p_ij * (1.0 - omega) + omega * (
												  ((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
												  rhs) / ((2.0 / (dx * dx)) + (2.0 / (dy * dy)));
			
			// calculate residual
			Real res = ((p_ip1j - (2.0 * p_ij) + p_im1j) / (dx * dx)) + ((p_ijp1 - (2.0 * p_ij) + p_ijm1) / (dy * dy)) - rhs;
			*norm_L2 += (res * res);
				
		} // end for row
	} // end for col
} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for black cells
 * 
 * \param[in]			F						array of discretized x-momentum eqn terms
 * \param[in]			G						array of discretized y-momentum eqn terms
 * \param[in]			pres_red		pressure values of red cells
 * \param[in]			dt					time-step size
 * \param[inout]	pres_black	pressure values of black cells
 * \param[inout]	norm_L2			variable holding summed residuals
 */
void black_kernel (const Real dt, const Real * F, const Real * G, const Real * pres_red,
									 Real * pres_black, Real * norm_L2)
{
	// loop over all cells
	for (uint col = 0; col < NUM; ++col) {
		for (uint row = 0; row < (NUM / 2); ++row) {
			
			uint ind_black = (col * (NUM >> 1)) + row;  						// local (black) index
			uint ind = (col * NUM) + (2 * row) + ((col + 1) & 1);	// global index
			
			Real p_ij = pres_black[ind_black];
			
			// left boundary
			Real p_im1j;
			Real F_im1j;
			if (col == 0) {
				p_im1j = p_ij;
				F_im1j = 0.0;
			} else {
				p_im1j = pres_red[((col - 1) * (NUM >> 1)) + row];
				F_im1j = F[((col - 1) * NUM) + (2 * row) + ((col + 1) & 1)];
			}
			
			// right boundary
			Real p_ip1j;
			if (col == (NUM - 1)) {
				p_ip1j = p_ij;
			} else {
				p_ip1j = pres_red[((col + 1) * (NUM >> 1)) + row];
			}
			
			// bottom boundary
			Real p_ijm1;
			Real G_ijm1;
			if (((2 * row) + ((col + 1) & 1)) == 0) {
				p_ijm1 = p_ij;
				G_ijm1 = 0.0;
			} else {
				p_ijm1 = pres_red[(col * (NUM >> 1)) + row - (col & 1)];
				G_ijm1 = G[(col * NUM) + (2 * row) + ((col + 1) & 1) - 1];
			}
			
			// top boundary
			Real p_ijp1;
			if (((2 * row) + ((col + 1) & 1)) == (NUM - 1)) {
				p_ijp1 = p_ij;
			} else {
				p_ijp1 = pres_red[(col * (NUM >> 1)) + row + ((col + 1) & 1)];
			}
			
			// right-hand side
			Real rhs = (((F[ind] - F_im1j) / dx) + ((G[ind] - G_ijm1) / dy)) / dt;
			
			pres_black[ind_black] = p_ij * (1.0 - omega) + omega * (
												  		((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
												  		rhs) / ((2.0 / (dx * dx)) + (2.0 / (dy * dy)));
			
			// calculate residual
			Real res = ((p_ip1j - (2.0 * p_ij) + p_im1j) / (dx * dx)) + ((p_ijp1 - (2.0 * p_ij) + p_ijm1) / (dy * dy)) - rhs;
			*norm_L2 += (res * res);
				
		} // end for row
	} // end for col
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

void calculate_u (const Real dt, const Real * F, 
									const Real * pres_red, const Real * pres_black, 
									Real * u)
{
	for (uint col = 0; col < NUM - 1; ++col) {
		for (uint row = 0; row < NUM; ++row) {
			
			uint ind = (col * NUM) + row;
			
			Real p_ij, p_ip1j;
			if (((row + col) & 1) == 0) {
				// red pressure cell
				p_ij = pres_red[(col * (NUM >> 1)) + ((row - (col & 1)) >> 1)];
				
				// p_ip1j is black cell
				p_ip1j = pres_black[((col + 1) * (NUM >> 1)) + ((row - (col & 1)) >> 1)];
			} else {
				// black pressure cell
				p_ij = pres_black[(col * (NUM >> 1)) + ((row - ((col + 1) & 1)) >> 1)];
				
				// p_ip1j is red cell
				p_ip1j = pres_red[((col + 1) * (NUM >> 1)) + ((row - ((col + 1) & 1)) >> 1)];
			}
			
			u[ind] = F[ind] - (dt * (p_ip1j - p_ij) / dx);
			
		} // end for row
	} // end for col
	
} // end calculate_u

///////////////////////////////////////////////////////////////////////////////

void calculate_v (const Real dt, const Real * G, 
									const Real * pres_red, const Real * pres_black, 
									Real * v)
{
	for (uint col = 0; col < NUM; ++col) {
		for (uint row = 0; row < NUM - 1; ++row) {
			
			uint ind = (col * NUM) + row;
			
			Real p_ij, p_ijp1;
			if (((row + col) & 1) == 0) {
				// red pressure cell
				//p_ij = pres_red[(col * (NUM >> 1)) + ((row - (col & 1)) >> 1)];
				p_ij = pres_red[(col * (NUM / 2)) + ((row - (col % 2)) / 2)];
				
				// p_ijp1 is black cell
				//p_ijp1 = pres_black[(col * (NUM >> 1)) + ((row + 1 - ((col + 1) & 1)) >> 1)];
				p_ijp1 = pres_black[(col * (NUM / 2)) + ((row + 1 - ((col + 1) % 2)) / 2)];
			} else {
				// black pressure cell
				//p_ij = pres_black[(col * (NUM >> 1)) + ((row - ((col + 1) & 1)) >> 1)];
				p_ij = pres_black[(col * (NUM / 2)) + ((row - ((col + 1) % 2)) / 2)];
				
				// p_ijp1 is red cell
				//p_ijp1 = pres_red[(col * (NUM >> 1)) + ((row + 1 - (col & 1)) >> 1)];
				p_ijp1 = pres_red[(col * (NUM / 2)) + ((row + 1 - (col % 2)) / 2)];
			}
			
			v[ind] = G[ind] - (dt * (p_ijp1 - p_ij) / dy);
			
		} // end for row
	} // end for col
	
} // end calculate_v

///////////////////////////////////////////////////////////////////////////////

int main (void)
{
	
	
	// iterations for Red-Black Gauss-Seidel with SOR
	uint iter = 0;
	uint it_max = 10000;
	
	// SOR iteration tolerance
	Real tol = 0.001;
	
	// time range
	Real time_start = 0.0;
	Real time_end = 20.0;
	
	// initial time step size
	Real dt = 0.02;
	
	uint size = NUM * NUM;
	uint size_pres = (NUM / 2) * NUM;
	
	// arrays for pressure and velocity
	Real *F, *u;
	Real *G, *v;
	
	F = (Real *) calloc (size, sizeof(Real));
	u = (Real *) calloc (size, sizeof(Real));
	G = (Real *) calloc (size, sizeof(Real));
	v = (Real *) calloc (size, sizeof(Real));
	
	for (uint i = 0; i < size; ++i) {
		F[i] = 0.0;
		u[i] = 0.0;
		G[i] = 0.0;
		v[i] = 0.0;
	}
	
	// arrays for pressure
	Real *pres_red, *pres_black;
	
	pres_red = (Real *) calloc (size_pres, sizeof(Real));
	pres_black = (Real *) calloc (size_pres, sizeof(Real));
	
	for (uint i = 0; i < size_pres; ++i) {
		pres_red[i] = 0.0;
		pres_black[i] = 0.0;
	}
	
	//////////////////////////////
	// start timer
	clock_t start_time = clock();
	//////////////////////////////
	
	
	Real time = time_start;
	
	// time iteration loop
	while (time < time_end) {
		
		// increase time
		time += dt;
		
		// calculate F and G
		
		calculate_F (u, v, dt, F);
		calculate_G (u, v, dt, G);
		
		// calculate new pressure
		// red-black Gauss-Seidel with SOR iteration loop
		for (iter = 1; iter <= it_max; ++iter) {

			Real norm_L2 = 0.0;

			// update red cells
			red_kernel (dt, F, G, pres_black, pres_red, &norm_L2);

			// update black cells
			black_kernel (dt, F, G, pres_red, pres_black, &norm_L2);

			// calculate residual
			norm_L2 = sqrt(norm_L2 / ((double)size));

			// if tolerance has been reached, end SOR iterations
			if (norm_L2 < tol) {
				break;
			}	
		} // end for
		
		
		// calculate new u and v velocities
		calculate_u (dt, F, pres_red, pres_black, u);
		calculate_v (dt, G, pres_red, pres_black, v);
		
		// calculate new time step based on stability
		dt = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));
		
		Real max_u = 0.0;
		for (uint i = 0; i < NUM * NUM; ++i) {
			Real test_u = fabs(u[i]);
			max_u = MAX(max_u, test_u);
		}
		
		Real max_v = 0.0;
		for (uint i = 0; i < NUM * NUM; ++i) {
			Real test_v = fabs(v[i]);
			max_v = MAX(max_v, test_v);
		}
		
		max_u = MIN((dx / max_u), (dy / max_v));
		dt = tau * MIN(dt, max_u);
		
		if ((time + dt) >= time_end) {
			dt = time_end - time;
		}
		
		printf("Time: %f, iterations: %i\n", time, iter);
		
		
		//exit(1);
		
		
		
	} // end while
	
	/////////////////////////////////
	// end timer
	clock_t end_time = clock();
	/////////////////////////////////
	
	printf("CPU:\n");
	printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	
	// write data to file
	FILE * pfile;
	pfile = fopen("velocity_cpu.dat", "w");
	fprintf(pfile, "#x\ty\tu\tv\n");
	if (pfile != NULL) {
		for (uint row = 0; row < NUM; ++row) {
			for (uint col = 0; col < NUM; ++col) {
				
				Real u_ij = u[(col * NUM) + row];
				Real u_im1j;
				if (col == 0) {
					u_im1j = 0.0;
				} else {
					u_im1j = u[(col - 1) * NUM + row];
				}
				
				u_ij = (u_ij + u_im1j) / 2.0;
				
				Real v_ij = v[(col * NUM) + row];
				Real v_ijm1;
				if (row == 0) {
					v_ijm1 = 0.0;
				} else {
					v_ijm1 = v[(col * NUM) + row - 1];
				}
				
				v_ij = (v_ij + v_ijm1) / 2.0;
				
				fprintf(pfile, "%f\t%f\t%f\t%f\n", ((double)col + 0.5) * dx, ((double)row + 0.5) * dy, u_ij, v_ij);
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