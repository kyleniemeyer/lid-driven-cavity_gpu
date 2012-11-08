/** GPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_gpu.cu
 *
 * \author Kyle E. Niemeyer
 * \date 09/27/2012
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity on a GPU (via CUDA), using the finite difference method.
 * To change the grid resolution, modify "NUM". In addition, the problem is controlled
 * by the Reynolds number ("Re_num").
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

// CUDA libraries
#include <cuda.h>
#include <cutil.h>

/** Problem size along one side; total number of cells is this squared */
#define NUM 256

// block size
#define BLOCK_SIZE 64

/** Double precision */
//#define DOUBLE

#ifdef DOUBLE
	#define Real double
	
	#define ZERO 0.0
	#define ONE 1.0
	#define TWO 2.0
	#define FOUR 4.0
#else
	#define Real float
	// replace double functions with float versions
	#define fmin fminf
	#define fmax fmaxf
	#define fabs fabsf
	
	#define ZERO 0.0f
	#define ONE 1.0f
	#define TWO 2.0f
	#define FOUR 4.0f
#endif

/** Use shared memory */
#define SHARED

/** Use atomic operations to calculate residual, only for SINGLE PRECISION */
//#define ATOMIC

#if defined (ATOMIC) && defined (DOUBLE)
# error double precision atomic operations not supported
#endif

typedef unsigned int uint;

/** Reynolds number */
#ifdef DOUBLE
const Real Re_num = 1000.0;
#else
const Real Re_num = 1000.0f;
#endif

/** SOR relaxation parameter */
#ifdef DOUBLE
const Real omega = 1.7;
#else
const Real omega = 1.7f;
#endif

/** Discretization mixture parameter */
#ifdef DOUBLE
const Real mix_param = 0.9;
#else
const Real mix_param = 0.9f;
#endif

/** Safety factor for time step modification */
#ifdef DOUBLE
const Real tau = 0.5;
#else
const Real tau = 0.5f;
#endif

/** Body forces in x- and y- directions */
const Real gx = ZERO;
const Real gy = ZERO;

/** Normalized lid velocity */
const Real uStar = ONE;

/** Domain size (non-dimensional) */
#define xLength ONE
#define yLength ONE

/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

/** Max macro (type safe, from GNU) */
#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

#ifdef ATOMIC
// need bijection between float and unsigned int to use atomicMax()

static __inline__ __device__ unsigned int floatFlip (float theFloat)
{
	unsigned int mask = (__float_as_int(theFloat) >> 31) | 0x80000000;
	return __float_as_int(theFloat) ˆ mask;
}

inline __host__ float invFloatFlip (unsigned int theUint)
{
	unsigned int mask = ((theUint >> 31) - 1) | 0x80000000;
	return __int_as_float((int)(theUint ^ mask));
}
#endif

///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_F (const Real * u, const Real * v, const Real dt, 
									 Real * F)
{	
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (col == (NUM - 1)) {
		// right boundary, F_ij = u_ij
		F[((NUM - 1) * NUM) + row] = u[((NUM - 1) * NUM) + row];
	} else {
		
		// u and v velocities
		Real u_ij = u[(col * NUM) + row];
		Real u_ip1j = u[((col + 1) * NUM) + row];
		
		Real v_ij = v[(col * NUM) + row];
		Real v_ip1j = v[((col + 1) * NUM) + row];
		
		// left boundary
		Real u_im1j;
		if (col == 0) {
			u_im1j = ZERO;
		} else {
			u_im1j = u[((col - 1) * NUM) + row];
		}
		
		// bottom boundary
		Real u_ijm1, v_ijm1, v_ip1jm1;
		if (row == 0) {
			u_ijm1 = -u_ij;
			v_ijm1 = ZERO;
			v_ip1jm1 = ZERO;
		} else {
			u_ijm1 = u[(col * NUM) + row - 1];
			v_ijm1 = v[(col * NUM) + row - 1];
			v_ip1jm1 = v[((col + 1) * NUM) + row - 1];
		}
		
		// top boundary
		Real u_ijp1;
		if (row == (NUM - 1)) {
			u_ijp1 = (TWO * uStar) - u_ij;
		} else {
			u_ijp1 = u[(col * NUM) + row + 1];
		}
		
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

		F[(col * NUM) + row] = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
		
	} // end if
		
} // end calculate_F

///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_G (const Real * u, const Real * v, const Real dt, 
									 Real * G)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (row == (NUM - 1)) {
		G[(col * NUM) + NUM - 1] = v[(col * NUM) + NUM - 1];
	} else {
		
		// u and v velocities
		Real u_ij = u[(col * NUM) + row];
		Real u_ijp1 = u[(col * NUM) + row + 1];
		
		Real v_ij = v[(col * NUM) + row];
		Real v_ijp1 = v[(col * NUM) + row + 1];
		
		// bottom boundary
		Real v_ijm1;
		if (row == 0) {
			v_ijm1 = ZERO;
		} else {
			v_ijm1 = v[(col * NUM) + row - 1];
		}
		
		// left boundary
		Real v_im1j, u_im1j, u_im1jp1;
		if (col == 0) {
			v_im1j = -v_ij;
			u_im1j = ZERO;
			u_im1jp1 = ZERO;
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
					- fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (FOUR * dy);
		duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
					+ mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
					- fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (FOUR * dx);
	  	d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
	  	d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);

		G[(col * NUM) + row] = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
			
	} // end if
		
} // end calculate_G

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
__global__ 
void red_kernel (const Real dt, const Real * F, const Real * G, const Real * pres_black,
								 Real * pres_red, Real * norm_L2)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	int ind_red = (col * (NUM >> 1)) + row;  					// local (red) index
	int ind = (col * NUM) + (2 * row) + (col & 1);		// global index

	Real p_ij = pres_red[ind_red];			

	// left boundary
	Real p_im1j;
	Real F_im1j;
	if (col == 0) {
		p_im1j = p_ij;
		F_im1j = ZERO;
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
		G_ijm1 = ZERO;
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

	pres_red[ind_red] = p_ij * (ONE - omega) + omega * (
										  ((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
										  rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));
	
	// calculate residual (reuse rhs variable)
	rhs = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx)) + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;
	
	#ifdef SHARED
		// store residual for block
		__shared__ Real res_cache[BLOCK_SIZE];
	
		// store squared residual from each thread
		res_cache[threadIdx.y] = rhs * rhs;
	
		// synchronize threads in block
		__syncthreads();
	
		// add up squared residuals for block
		int i = BLOCK_SIZE >> 1;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
			}
			__syncthreads();
			i >>= 1;
		}
	
		// store block's summed residuals
		if (threadIdx.y == 0) {
			#ifdef ATOMIC
				atomicAdd (norm_L2, res_cache[0]);
			#else
				norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
			#endif
		}
	#else
		norm_L2[ind_red] = rhs * rhs;
	#endif
	
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
__global__ 
void black_kernel (const Real dt, const Real * F, const Real * G, const Real * pres_red,
									 Real * pres_black, Real * norm_L2)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	int ind_black = (col * (NUM >> 1)) + row;  						// local (black) index
	int ind = (col * NUM) + (2 * row) + ((col + 1) & 1);	// global index
	
	Real p_ij = pres_black[ind_black];
	
	// left boundary
	Real p_im1j;
	Real F_im1j;
	if (col == 0) {
		p_im1j = p_ij;
		F_im1j = ZERO;
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
		G_ijm1 = ZERO;
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
	
	pres_black[ind_black] = p_ij * (ONE - omega) + omega * (
										  		((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
										  		rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));
	
	// calculate residual (reuse rhs variable)
	rhs = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx)) + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;
	
	#ifdef SHARED
		// store residual for block
		__shared__ Real res_cache[BLOCK_SIZE];
	
		// store squared residual from each thread
		res_cache[threadIdx.y] = rhs * rhs;
	
		// synchronize threads in block
		__syncthreads();
	
		// add up squared residuals for block
		int i = BLOCK_SIZE >> 1;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
			}
			__syncthreads();
			i >>= 1;
		}
	
		// store block's summed residuals
		if (threadIdx.y == 0) {
			#ifdef ATOMIC
				atomicAdd (norm_L2, res_cache[0]);
			#else
				norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
			#endif
		}
	#else
		norm_L2[ind_black] = rhs * rhs;
	#endif
	
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_u (const Real dt, const Real * F, 
									const Real * pres_red, const Real * pres_black, 
									#ifdef ATOMIC
									Real * u, unsigned int * max_u_d)
									#else
									Real * u, Real * max_u_d)
									#endif
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	#ifdef SHARED
		#ifdef ATOMIC
			__shared__ unsigned int res_cache[BLOCK_SIZE];
			res_cache[threadIdx.y] = 0;
		#else
			__shared__ Real res_cache[BLOCK_SIZE];
			res_cache[threadIdx.y] = ZERO;
		#endif
	#endif
	
	if (col != (NUM - 1)) {
		int ind = (col * NUM) + row;
		
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
		
		//u[ind] = F[ind] - (dt * (p_ip1j - p_ij) / dx);
		Real u_ij = F[ind] - (dt * (p_ip1j - p_ij) / dx);
		
		u[ind] = u_ij;
		
		#ifdef SHARED
		// store maximum u for block from each thread
			#ifdef ATOMIC
				res_cache[threadIdx.y] = floatFlip (fabs(u_ij));
			#else
				res_cache[threadIdx.y] = fabs(u_ij);
			#endif
		
		// synchronize threads in block
		__syncthreads();

		// add up squared residuals for block
		int i = BLOCK_SIZE >> 1;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] = fmax(res_cache[threadIdx.y], res_cache[threadIdx.y + i]);
			}
			__syncthreads();
			i >>= 1;
		}

		// store block's summed residuals
		if (threadIdx.y == 0) {
			#ifdef ATOMIC
				atomicMax (max_u_d, res_cache[0]);
			#else
				max_u_d[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
			#endif
		}
		#endif
	} // end if
	
} // end calculate_u

///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_v (const Real dt, const Real * G, 
									const Real * pres_red, const Real * pres_black, 
									#ifdef ATOMIC
									Real * v, unsigned int * max_v_d)
									#else
									Real * v, Real * max_v_d)
									#endif
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	#ifdef SHARED
		#ifdef ATOMIC
			__shared__ unsigned int res_cache[BLOCK_SIZE];
			res_cache[threadIdx.y] = 0;
		#else
			__shared__ Real res_cache[BLOCK_SIZE];
			res_cache[threadIdx.y] = ZERO;
		#endif
	#endif
	
	if (row != (NUM - 1)) {
		int ind = (col * NUM) + row;
		
		Real p_ij, p_ijp1;
		if (((row + col) & 1) == 0) {
			// red pressure cell
			p_ij = pres_red[(col * (NUM >> 1)) + ((row - (col & 1)) >> 1)];
			
			// p_ijp1 is black cell
			p_ijp1 = pres_black[(col * (NUM >> 1)) + ((row + 1 - ((col + 1) & 1)) >> 1)];
		} else {
			// black pressure cell
			p_ij = pres_black[(col * (NUM >> 1)) + ((row - ((col + 1) & 1)) >> 1)];
			
			// p_ijp1 is red cell
			p_ijp1 = pres_red[(col * (NUM >> 1)) + ((row + 1 - (col & 1)) >> 1)];
		}
		
		//v[ind] = G[ind] - (dt * (p_ijp1 - p_ij) / dy);
		Real v_ij = G[ind] - (dt * (p_ijp1 - p_ij) / dy);
		
		v[ind] = v_ij;
		
		#ifdef SHARED
		// store maximum v for block for each thread
		
			#ifdef ATOMIC
				res_cache[threadIdx.y] = floatFlip (fabs(v_ij));
			#else
				res_cache[threadIdx.y] = fabs(v_ij);
			#endif
		
		// synchronize threads in block
		__syncthreads();

		// add up squared residuals for block
		int i = BLOCK_SIZE >> 1;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] = fmax(res_cache[threadIdx.y], res_cache[threadIdx.y + i]);
			}
			__syncthreads();
			i >>= 1;
		}

		// store block's summed residuals
		if (threadIdx.y == 0) {
			#ifdef ATOMIC
				atomicMax (max_v_d, res_cache[0]);
			#else
				max_v_d[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
			#endif
		}
		#endif
	} // end if
	
} // end calculate_v

///////////////////////////////////////////////////////////////////////////////

int main (void)
{
	// iterations for Red-Black Gauss-Seidel with SOR
	uint iter = 0;
	const uint it_max = 10000;
	
	// SOR iteration tolerance
	const Real tol = 0.001;
	
	// time range
	const Real time_start = 0.0;
	const Real time_end = 20.0;
	
	// initial time step size
	Real dt = 0.02;
	
	uint size = NUM * NUM;
	uint size_pres = (NUM / 2) * NUM;
	
	// arrays for pressure and velocity
	Real *F, *u;
	Real *G, *v;
	
	F = (Real *) calloc ((NUM + 1) * NUM, sizeof(Real));
	u = (Real *) calloc ((NUM + 1) * NUM, sizeof(Real));
	G = (Real *) calloc ((NUM + 1) * NUM, sizeof(Real));
	v = (Real *) calloc ((NUM + 1) * NUM, sizeof(Real));
	
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
	
	////////////////////////////////////////
	// allocate and transfer device memory
	Real *u_d, *F_d, *v_d, *G_d;
	Real *pres_red_d, *pres_black_d;
	
	CUDA_SAFE_CALL (cudaMalloc ((void**) &u_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &F_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &v_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &G_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &pres_red_d, size_pres * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &pres_black_d, size_pres * sizeof(Real)));
	
	// copy to device memory
	CUDA_SAFE_CALL (cudaMemcpy (u_d, u, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (F_d, F, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (v_d, v, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (G_d, G, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (pres_red_d, pres_red, size_pres * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (pres_black_d, pres_black, size_pres * sizeof(Real), cudaMemcpyHostToDevice));
	
	////////////////////////////////////////
	// block and grid dimensions
	
	// block and grid dimensions for u and F
	dim3 dimBlock_u (1, BLOCK_SIZE);
	dim3 dimGrid_u (NUM, NUM / BLOCK_SIZE);
	
	// block and grid dimensions for v and G
	dim3 dimBlock_v (1, BLOCK_SIZE);
	dim3 dimGrid_v (NUM, NUM / BLOCK_SIZE);
	
	// block and grid dimensions for pressure
	dim3 dimBlock_p (1, BLOCK_SIZE);
	dim3 dimGrid_p (NUM, NUM / (2 * BLOCK_SIZE));
	
	// residual variable
	Real *res, *res_d;
	#ifdef SHARED
		#ifdef ATOMIC
			uint size_res = 1;
		#else
			uint size_res = dimGrid_p.x * dimGrid_p.y;
		#endif
	#else
		uint size_res = size_pres;
	#endif
	res = (Real *) malloc (size_res * sizeof(Real));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &res_d, size_res * sizeof(Real)));
	
	// variables to store maximum velocities
	
	#ifdef SHARED
		#ifdef ATOMIC
			unsigned int *max_u_arr, *max_v_arr;
			unsigned int *max_u_d, *max_v_d;
			uint size_max = 1;
	
			max_u_arr = (unsigned int *) malloc (size_max * sizeof(unsigned int));
			CUDA_SAFE_CALL (cudaMalloc ((void**) &max_u_d, size_max * sizeof(unsigned int)));
			max_v_arr = (unsigned int *) malloc (size_max * sizeof(unsigned int));
			CUDA_SAFE_CALL (cudaMalloc ((void**) &max_v_d, size_max * sizeof(unsigned int)));
		#else
			Real *max_u_d, *max_v_d;
			Real *max_u_arr, *max_v_arr;
			uint size_max = dimGrid_u.x * dimGrid_u.y;
	
			max_u_arr = (Real *) malloc (size_max * sizeof(Real));
			CUDA_SAFE_CALL (cudaMalloc ((void**) &max_u_d, size_max * sizeof(Real)));
			max_v_arr = (Real *) malloc (size_max * sizeof(Real));
			CUDA_SAFE_CALL (cudaMalloc ((void**) &max_v_d, size_max * sizeof(Real)));
		#endif
	#else
		Real *max_u_d, *max_v_d;
	#endif
	
	//////////////////////////////
	// start timer
	clock_t start_time = clock();
	//////////////////////////////
	
	Real time = time_start;
	
	// time-step size based on grid and Reynolds number
	Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));
	
	// time iteration loop
	while (time < time_end) {
		
		// increase time
		time += dt;
		
		// calculate F and G		
		calculate_F <<<dimGrid_u, dimBlock_u>>> (u_d, v_d, dt, F_d);
		calculate_G <<<dimGrid_v, dimBlock_v>>> (u_d, v_d, dt, G_d);
		
		
		// calculate new pressure
		// red-black Gauss-Seidel with SOR iteration loop
		for (iter = 1; iter <= it_max; ++iter) {

			Real norm_L2 = 0.0;
			
			#ifdef ATOMIC
				// set device value to zero
				*res = 0.0;
				CUDA_SAFE_CALL (cudaMemcpy (res_d, res, sizeof(Real), cudaMemcpyHostToDevice));
			#endif

			// update red cells
			red_kernel <<<dimGrid_p, dimBlock_p>>> (dt, F_d, G_d, pres_black_d, pres_red_d, res_d);
			
			#ifndef ATOMIC
				// transfer residual value(s) back to CPU and add red cell contributions
				CUDA_SAFE_CALL (cudaMemcpy (res, res_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost));
				for (uint i = 0; i < size_res; ++i) {
					norm_L2 += res[i];
				}
			#endif
			
			// update black cells
			black_kernel <<<dimGrid_p, dimBlock_p>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);
			
			// transfer residual value(s) back to CPU and add black cell contributions
			CUDA_SAFE_CALL (cudaMemcpy (res, res_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost));
			#ifdef ATOMIC
				norm_L2 = *res;
			#else
				for (uint i = 0; i < size_res; ++i) {
					norm_L2 += res[i];
				}
			#endif
			
			// calculate residual
			norm_L2 = sqrt(norm_L2 / ((Real)size));
			
			// if tolerance has been reached, end SOR iterations
			if (norm_L2 < tol) {
				break;
			}	
		} // end for
		
		// calculate new u and v velocities
		calculate_u <<<dimGrid_u, dimBlock_u>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
		calculate_v <<<dimGrid_v, dimBlock_v>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);
		
		// calculate new time step based on stability and CFL
		
		// need maximum u- and v- velocities
		Real max_v = 0.0;
		Real max_u = 0.0;
		
		#ifdef SHARED
			#ifdef ATOMIC
				CUDA_SAFE_CALL (cudaMemcpy (max_u_arr, max_u_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL (cudaMemcpy (max_v_arr, max_v_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			
				max_u = invFloatFlip (*max_u_arr);
				max_v = invFloatFlip (*max_v_arr);
			#else
				CUDA_SAFE_CALL (cudaMemcpy (max_u_arr, max_u_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL (cudaMemcpy (max_v_arr, max_v_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost));
			
				for (uint i = 0; i < size_max; ++i) {
					Real test_u = max_u_arr[i];
					max_u = MAX(max_u, test_u);

					Real test_v = max_v_arr[i];
					max_v = MAX(max_v, test_v);
				}
			#endif
		#else
			// transfer velocities back to CPU
			CUDA_SAFE_CALL (cudaMemcpy (u, u_d, size * sizeof(Real), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL (cudaMemcpy (v, v_d, size * sizeof(Real), cudaMemcpyDeviceToHost));
		
			for (uint i = 0; i < NUM * NUM; ++i) {
				Real test_u = fabs(u[i]);
				max_u = MAX(max_u, test_u);
			
				Real test_v = fabs(v[i]);
				max_v = MAX(max_v, test_v);
			}
		#endif
		
		max_u = MIN((dx / max_u), (dy / max_v));
		dt = tau * MIN(dt_Re, max_u);
		
		if ((time + dt) >= time_end) {
			dt = time_end - time;
		}
		
		printf("Time: %f, iterations: %i\n", time, iter);
		
	} // end while
	
	// transfer final temperature values back
	CUDA_SAFE_CALL (cudaMemcpy (u, u_d, size * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL (cudaMemcpy (v, v_d, size * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL (cudaMemcpy (pres_red, pres_red_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL (cudaMemcpy (pres_black, pres_red_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost));
	
	// free device memory
	CUDA_SAFE_CALL (cudaFree(u_d));
	CUDA_SAFE_CALL (cudaFree(v_d));
	CUDA_SAFE_CALL (cudaFree(F_d));
	CUDA_SAFE_CALL (cudaFree(G_d));
	CUDA_SAFE_CALL (cudaFree(pres_red_d));
	CUDA_SAFE_CALL (cudaFree(pres_black_d));
	
	#ifdef SHARE
		CUDA_SAFE_CALL (cudaFree(max_u_d));
		CUDA_SAFE_CALL (cudaFree(max_v_d));
	#endif
	
	/////////////////////////////////
	// end timer
	clock_t end_time = clock();
	/////////////////////////////////
	
	printf("GPU:\n");
	printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	
	// write data to file
	FILE * pfile;
	pfile = fopen("velocity_gpu.dat", "w");
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
	
	#ifdef SHARED
		free(max_u_arr);
		free(max_v_arr);
	#endif
	
	cudaDeviceReset();
	
	return 0;
}
