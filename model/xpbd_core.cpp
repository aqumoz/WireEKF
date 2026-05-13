/*
 * xpbd_core.cpp
 * =============
 * C++ implementation of the XPBD Cosserat-rod wire simulator inner loop.
 *
 * Exposes a single class  WireSimulatorCPP  with one method:
 *   estimate_wire_state(x, v, ee_pos, ee_quat, f_ext)
 *       -> (x_new, v_new, CW, ee_wrench)
 *
 * All arrays are passed as numpy float64 arrays.
 * Quaternion convention throughout: [x, y, z, w]  (scalar last).
 *
 * Build
 * -----
 *   pip install pybind11 numpy
 *   python setup.py build_ext --inplace
 *
 *   Or with CMake:
 *   cmake -B build && cmake --build build
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using arr    = py::array_t<double>;
using carr   = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ════════════════════════════════════════════════════════════════════════════
// Inline math utilities
// ════════════════════════════════════════════════════════════════════════════

// Quaternion [x,y,z,w] product: q1 * q2
static inline void qmul(const double* q1, const double* q2, double* out) {
    double x1=q1[0], y1=q1[1], z1=q1[2], w1=q1[3];
    double x2=q2[0], y2=q2[1], z2=q2[2], w2=q2[3];
    out[0] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    out[1] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    out[2] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
    out[3] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
}

// Quaternion conjugate [-x,-y,-z,w]
static inline void qconj(const double* q, double* out) {
    out[0]=-q[0]; out[1]=-q[1]; out[2]=-q[2]; out[3]=q[3];
}

// 3×3 rotation matrix from quaternion [x,y,z,w]  (row-major)
static inline void qrotm(const double* q, double* R) {
    double x=q[0], y=q[1], z=q[2], w=q[3];
    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

// R (row-major 3×3) @ v3 → out3
static inline void Rv(const double* R, const double* v, double* out) {
    for(int i=0;i<3;i++) out[i]=R[3*i]*v[0]+R[3*i+1]*v[1]+R[3*i+2]*v[2];
}
// R^T @ v3 → out3
static inline void RTv(const double* R, const double* v, double* out) {
    for(int i=0;i<3;i++) out[i]=R[i]*v[0]+R[3+i]*v[1]+R[6+i]*v[2];
}

static inline double dot3(const double* a, const double* b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static inline double dot4(const double* a, const double* b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
}
static inline void cross3(const double* a, const double* b, double* out) {
    out[0]=a[1]*b[2]-a[2]*b[1];
    out[1]=a[2]*b[0]-a[0]*b[2];
    out[2]=a[0]*b[1]-a[1]*b[0];
}
static inline void normalise4(double* v) {
    double n=std::sqrt(dot4(v,v));
    v[0]/=n; v[1]/=n; v[2]/=n; v[3]/=n;
}

// Solve 3×3 diagonal linear system A*x=b  (A stored as 9-element row-major)
// For our use case (isotropic I) this is just scalar division.
// We keep the general path for correctness.
static inline void solve3(const double* A, const double* b, double* x) {
    // Cramer / Gaussian elim (3x3, pivoting omitted — inertia matrices are SPD)
    double a00=A[0],a01=A[1],a02=A[2];
    double a10=A[3],a11=A[4],a12=A[5];
    double a20=A[6],a21=A[7],a22=A[8];
    double det = a00*(a11*a22-a12*a21)
               - a01*(a10*a22-a12*a20)
               + a02*(a10*a21-a11*a20);
    double inv = 1.0/det;
    x[0] = inv*((a11*a22-a12*a21)*b[0]-(a01*a22-a02*a21)*b[1]+(a01*a12-a02*a11)*b[2]);
    x[1] = inv*(-(a10*a22-a12*a20)*b[0]+(a00*a22-a02*a20)*b[1]-(a00*a12-a02*a10)*b[2]);
    x[2] = inv*((a10*a21-a11*a20)*b[0]-(a00*a21-a01*a20)*b[1]+(a00*a11-a01*a10)*b[2]);
}

// ════════════════════════════════════════════════════════════════════════════
// Constraint solvers  (mirroring constraints.py exactly)
// ════════════════════════════════════════════════════════════════════════════

// Stretch / shear — updates corrections in-place
static void solve_stretch_shear(
    const double* p0, double im0,
    const double* p1, double im1,
    const double* q0, double im_q0,
    double L,
    double*       lam,           // (3,) accumulated λ
    const double* alpha,         // (3,) compliance
    const double* C_n,           // (3,) violation at step start
    const double* gamma,         // (3,) Rayleigh γ
    double        dt,
    double* c0, double* c1, double* cq0, double* dlam)
{
    double qx=q0[0], qy=q0[1], qz=q0[2], qw=q0[3];

    // d3 = local z-axis in world frame
    double d3[3];
    d3[0]=2*(qx*qz+qw*qy);
    d3[1]=2*(qy*qz-qw*qx);
    d3[2]=qw*qw-qx*qx-qy*qy+qz*qz;

    double R[9]; qrotm(q0, R);

    // C_glob = (p1-p0)/L - d3
    double C_glob[3];
    for(int i=0;i<3;i++) C_glob[i]=(p1[i]-p0[i])/L - d3[i];

    // C_loc = R^T @ C_glob
    double C_loc[3]; RTv(R, C_glob, C_loc);

    // effective inverse mass
    double w = (im0+im1)/L + im_q0*4.0*L;

    // XPBD update
    double dt2 = dt*dt;
    for(int k=0;k<3;k++){
        double at = alpha[k]/dt2;
        double num = -C_loc[k] - at*lam[k] - gamma[k]*(C_loc[k]-C_n[k]);
        double den = (1.0+gamma[k])*w + at + 1e-9;
        dlam[k]   = num/den;
    }

    // gamma_loc = -dlam,  gamma_glob = R @ gamma_loc
    double g_loc[3]={ -dlam[0], -dlam[1], -dlam[2] };
    double g_glob[3]; Rv(R, g_loc, g_glob);

    for(int i=0;i<3;i++){ c0[i]= im0*g_glob[i]; c1[i]=-im1*g_glob[i]; }

    // quaternion correction
    double q_e3_bar[4]={ -qy, qx, -qw, qz };
    double g_quat[4]={ g_glob[0], g_glob[1], g_glob[2], 0.0 };
    double tmp[4]; qmul(g_quat, q_e3_bar, tmp);
    double sc = 2.0*im_q0*L;
    for(int i=0;i<4;i++) cq0[i]=tmp[i]*sc;
}

// Bend / twist
static void solve_bend_twist(
    const double* q0, double im_q0,
    const double* q1, double im_q1,
    const double* rest_darboux,   // (4,)
    double*       lam,
    const double* alpha,
    const double* C_n,
    const double* gamma,
    double        dt,
    double* cq0, double* cq1, double* dlam)
{
    // omega = qconj(q0) * q1
    double q0c[4]; qconj(q0, q0c);
    double omega[4]; qmul(q0c, q1, omega);

    // shorter-arc selection
    double op[4], om[4];
    for(int i=0;i<4;i++){ op[i]=omega[i]+rest_darboux[i]; om[i]=omega[i]-rest_darboux[i]; }
    if(dot4(om,om)>dot4(op,op))
        for(int i=0;i<4;i++) omega[i]=op[i];
    else
        for(int i=0;i<4;i++) omega[i]=om[i];

    const double* C_loc = omega;   // xyz part

    double dt2=dt*dt, imsum=im_q0+im_q1;
    for(int k=0;k<3;k++){
        double at = alpha[k]/dt2;
        double num = -C_loc[k] - at*lam[k] - gamma[k]*(C_loc[k]-C_n[k]);
        double den = (1.0+gamma[k])*imsum + at + 1e-9;
        dlam[k]   = num/den;
    }

    double corr[4]={ -dlam[0], -dlam[1], -dlam[2], 0.0 };
    double t0[4], t1[4];
    qmul(q1, corr, t0);
    qmul(q0, corr, t1);
    for(int i=0;i<4;i++){ cq0[i]= im_q0*t0[i]; cq1[i]=-im_q1*t1[i]; }
}

// EE position
static void solve_ee_position(
    const double* p0, double im0,
    const double* ee_pos,
    double* lam,
    double alpha_ee, double dt,
    const double* C_n, double gamma,
    double* corr, double* dlam)
{
    double at=alpha_ee/(dt*dt);
    for(int k=0;k<3;k++){
        double C=p0[k]-ee_pos[k];
        double num=-C - at*lam[k] - gamma*(C-C_n[k]);
        double den=(1.0+gamma)*im0 + at + 1e-9;
        dlam[k]=num/den;
        corr[k]=im0*dlam[k];
    }
}

// EE orientation
static void solve_ee_orient(
    const double* q0, double im_q0,
    const double* ee_quat,
    double* lam,
    double alpha_ee, double dt,
    const double* C_n, double gamma,
    double* corrq, double* dlam)
{
    double eec[4]; qconj(ee_quat, eec);
    double omega[4]; qmul(eec, q0, omega);

    // shorter-arc (rest = identity [0,0,0,1])
    double om_p[4]={ omega[0], omega[1], omega[2], omega[3]+1.0 };
    double om_m[4]={ omega[0], omega[1], omega[2], omega[3]-1.0 };
    if(dot4(om_m,om_m)>dot4(om_p,om_p))
        for(int i=0;i<4;i++) omega[i]=om_p[i];
    else
        for(int i=0;i<4;i++) omega[i]=om_m[i];

    double at=alpha_ee/(dt*dt);
    for(int k=0;k<3;k++){
        double num=-omega[k] - at*lam[k] - gamma*(omega[k]-C_n[k]);
        double den=(1.0+gamma)*im_q0 + at + 1e-9;
        dlam[k]=num/den;
    }
    double corr[4]={ -dlam[0], -dlam[1], -dlam[2], 0.0 };
    double tmp[4]; qmul(ee_quat, corr, tmp);
    for(int i=0;i<4;i++) corrq[i]=-im_q0*tmp[i];
}

// Constraint violations (for C^n precomputation)
static void stretch_shear_violation(
    const double* p0, const double* p1, const double* q0, double L, double* out)
{
    double qx=q0[0],qy=q0[1],qz=q0[2],qw=q0[3];
    double d3[3]={ 2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), qw*qw-qx*qx-qy*qy+qz*qz };
    double R[9]; qrotm(q0, R);
    double Cg[3]; for(int i=0;i<3;i++) Cg[i]=(p1[i]-p0[i])/L-d3[i];
    RTv(R, Cg, out);
}

static void bend_twist_violation(
    const double* q0, const double* q1, const double* rest, double* out)
{
    double q0c[4]; qconj(q0, q0c);
    double omega[4]; qmul(q0c, q1, omega);
    double op[4], om[4];
    for(int i=0;i<4;i++){ op[i]=omega[i]+rest[i]; om[i]=omega[i]-rest[i]; }
    const double* res=(dot4(om,om)>dot4(op,op))?op:om;
    out[0]=res[0]; out[1]=res[1]; out[2]=res[2];
}

// ════════════════════════════════════════════════════════════════════════════
// WireSimulatorCPP class
// ════════════════════════════════════════════════════════════════════════════

class WireSimulatorCPP {
public:
    int    N, dof, solver_iter;
    double L, dt;
    double alpha_ee_pos, alpha_ee_orient;
    double gamma_ee_pos, gamma_ee_orient;

    std::vector<double> alpha_stretch;  // (3,)
    std::vector<double> alpha_bend;     // (3,)
    std::vector<double> gamma_stretch;  // (3,)
    std::vector<double> gamma_bend;     // (3,)
    std::vector<double> m_num;          // (N,)
    std::vector<double> I_flat;         // (N*9,) row-major 3x3 per node
    std::vector<double> inv_I_flat;     // (N*9,) inverses

    WireSimulatorCPP(
        int N_, int dof_, double L_, double dt_, int solver_iter_,
        carr alpha_stretch_,  carr alpha_bend_,
        carr beta_stretch_,   carr beta_bend_,
        double alpha_ee_pos_, double alpha_ee_orient_,
        double beta_ee_pos_,  double beta_ee_orient_,
        carr m_num_,          carr I_flat_)   // I passed as (N*9,) flattened
    : N(N_), dof(dof_), L(L_), dt(dt_), solver_iter(solver_iter_),
      alpha_ee_pos(alpha_ee_pos_), alpha_ee_orient(alpha_ee_orient_)
    {
        auto as=alpha_stretch_.unchecked<1>(), ab=alpha_bend_.unchecked<1>();
        auto bs=beta_stretch_.unchecked<1>(),  bb=beta_bend_.unchecked<1>();
        auto mn=m_num_.unchecked<1>(),         If=I_flat_.unchecked<1>();

        alpha_stretch.resize(3); alpha_bend.resize(3);
        gamma_stretch.resize(3); gamma_bend.resize(3);
        for(int k=0;k<3;k++){
            alpha_stretch[k]=as(k); alpha_bend[k]=ab(k);
            gamma_stretch[k]=as(k)*bs(k)/dt;
            gamma_bend[k]   =ab(k)*bb(k)/dt;
        }
        gamma_ee_pos    = alpha_ee_pos_  * beta_ee_pos_  / dt;
        gamma_ee_orient = alpha_ee_orient_* beta_ee_orient_/ dt;

        m_num.resize(N);
        I_flat.resize(9*N); inv_I_flat.resize(9*N);
        for(int i=0;i<N;i++){
            m_num[i]=mn(i);
            for(int k=0;k<9;k++) I_flat[9*i+k]=If(9*i+k);
            // invert (SPD 3×3)
            double* A=&I_flat[9*i]; double* Ai=&inv_I_flat[9*i];
            double det=A[0]*(A[4]*A[8]-A[5]*A[7])
                      -A[1]*(A[3]*A[8]-A[5]*A[6])
                      +A[2]*(A[3]*A[7]-A[4]*A[6]);
            double id=1.0/det;
            Ai[0]= id*(A[4]*A[8]-A[5]*A[7]); Ai[1]=-id*(A[1]*A[8]-A[2]*A[7]); Ai[2]= id*(A[1]*A[5]-A[2]*A[4]);
            Ai[3]=-id*(A[3]*A[8]-A[5]*A[6]); Ai[4]= id*(A[0]*A[8]-A[2]*A[6]); Ai[5]=-id*(A[0]*A[5]-A[2]*A[3]);
            Ai[6]= id*(A[3]*A[7]-A[4]*A[6]); Ai[7]=-id*(A[0]*A[7]-A[1]*A[6]); Ai[8]= id*(A[0]*A[4]-A[1]*A[3]);
        }
    }

    // ── Main step ──────────────────────────────────────────────────────────

    py::tuple estimate_wire_state(
        carr x_init_, carr v_init_,
        carr ee_pos_, carr ee_quat_,
        carr f_ext_)
    {
        auto xi=x_init_.unchecked<1>(), vi=v_init_.unchecked<1>();
        auto ep=ee_pos_.unchecked<1>(), eq=ee_quat_.unchecked<1>();
        auto fe=f_ext_.unchecked<1>();

        const int DOF=dof;

        // Working copies
        std::vector<double> x(DOF*N), v(6*N);
        for(int i=0;i<DOF*N;i++) x[i]=xi(i);
        for(int i=0;i<6*N;i++)   v[i]=vi(i);

        double ee_pos[3]={ ep(0),ep(1),ep(2) };
        double ee_quat[4]={ eq(0),eq(1),eq(2),eq(3) };
        double rest_darboux[4]={ 0,0,0,1 };

        // Precompute inv_mass_rot per node (trace/3)
        std::vector<double> imr(N);
        for(int i=0;i<N;i++){
            double* I=&I_flat[9*i];
            imr[i]=3.0/(I[0]+I[4]+I[8]);
        }

        // ── 1. Prediction step ────────────────────────────────────────────
        for(int i=0;i<N;i++){
            int sx=i*DOF, sv=i*6;
            double im=1.0/m_num[i];

            // linear
            for(int k=0;k<3;k++) v[sv+k] += dt*fe(sv+k)*im;
            for(int k=0;k<3;k++) x[sx+k] += dt*v[sv+k];

            // angular: omega += dt * I^{-1} (tau - omega x I*omega)
            double* Ifull=&I_flat[9*i];
            double* Iinv =&inv_I_flat[9*i];
            double tau[3]={ fe(sv+3),fe(sv+4),fe(sv+5) };
            double omega[3]={ v[sv+3],v[sv+4],v[sv+5] };
            double Iom[3]; Rv(Ifull, omega, Iom);
            double cx[3];  cross3(omega, Iom, cx);
            double rhs[3]; for(int k=0;k<3;k++) rhs[k]=tau[k]-cx[k];
            double dom[3]; Rv(Iinv, rhs, dom);
            for(int k=0;k<3;k++) omega[k]+=dt*dom[k];
            for(int k=0;k<3;k++) v[sv+3+k]=omega[k];

            // quaternion integration
            double* q=&x[sx+3];
            double qom[4]={ omega[0],omega[1],omega[2],0.0 };
            double qdot[4]; qmul(q, qom, qdot);
            for(int k=0;k<4;k++) q[k]+=0.5*dt*qdot[k];
            normalise4(q);
        }

        // ── 2. Precompute C^n ─────────────────────────────────────────────
        std::vector<double> C_str_n(3*(N-1), 0.0);
        std::vector<double> C_ben_n(3*(N-1), 0.0);
        for(int i=1;i<N;i++){
            int s0=(i-1)*DOF, s1=i*DOF;
            stretch_shear_violation(&xi(s0),&xi(s1),&xi(s0+3), L, &C_str_n[3*(i-1)]);
            bend_twist_violation   (&xi(s0+3),&xi(s1+3), rest_darboux, &C_ben_n[3*(i-1)]);
        }
        double C_ee_pos_n[3],  C_ee_ori_n[3];
        for(int k=0;k<3;k++) C_ee_pos_n[k]=xi(k)-ee_pos[k];
        // EE orient violation
        {
            double eec[4]; qconj(ee_quat,eec);
            double om[4];  qmul(eec,&xi(3),om);
            double op[4]={ om[0],om[1],om[2],om[3]+1 };
            double omi[4]={ om[0],om[1],om[2],om[3]-1 };
            const double* r=(dot4(omi,omi)>dot4(op,op))?op:omi;
            for(int k=0;k<3;k++) C_ee_ori_n[k]=r[k];
        }

        // ── 3. Lagrange multipliers ───────────────────────────────────────
        std::vector<double> lam_str(3*(N-1),0), lam_ben(3*(N-1),0);
        double lam_ee_pos[3]={0,0,0}, lam_ee_ori[3]={0,0,0};

        // ── 4. Gauss-Seidel loop ──────────────────────────────────────────
        for(int iter=0;iter<solver_iter;iter++){

            // Stretch / shear
            for(int i=1;i<N;i++){
                int sp0=(i-1)*DOF, sp1=i*DOF, sq0=sp0+3;
                double c0[3],c1[3],cq0[4],dlam[3];
                solve_stretch_shear(
                    &x[sp0], 1.0/m_num[i-1],
                    &x[sp1], 1.0/m_num[i],
                    &x[sq0], imr[i-1],
                    L,
                    &lam_str[3*(i-1)],
                    alpha_stretch.data(),
                    &C_str_n[3*(i-1)],
                    gamma_stretch.data(),
                    dt, c0, c1, cq0, dlam);
                for(int k=0;k<3;k++) lam_str[3*(i-1)+k]+=dlam[k];
                for(int k=0;k<3;k++){ x[sp0+k]+=c0[k]; x[sp1+k]+=c1[k]; }
                for(int k=0;k<4;k++) x[sq0+k]+=cq0[k];
                normalise4(&x[sq0]);
            }

            // Bend / twist
            for(int i=1;i<N;i++){
                int sq0=(i-1)*DOF+3, sq1=i*DOF+3;
                double cq0[4],cq1[4],dlam[3];
                solve_bend_twist(
                    &x[sq0], imr[i-1],
                    &x[sq1], imr[i],
                    rest_darboux,
                    &lam_ben[3*(i-1)],
                    alpha_bend.data(),
                    &C_ben_n[3*(i-1)],
                    gamma_bend.data(),
                    dt, cq0, cq1, dlam);
                for(int k=0;k<3;k++) lam_ben[3*(i-1)+k]+=dlam[k];
                for(int k=0;k<4;k++){ x[sq0+k]+=cq0[k]; x[sq1+k]+=cq1[k]; }
                normalise4(&x[sq0]); normalise4(&x[sq1]);
            }

            // Normalise all quaternions
            for(int i=0;i<N;i++) normalise4(&x[i*DOF+3]);

            // EE position
            {
                double c[3],dlam[3];
                solve_ee_position(x.data(), 1.0/m_num[0], ee_pos,
                    lam_ee_pos, alpha_ee_pos, dt,
                    C_ee_pos_n, gamma_ee_pos, c, dlam);
                for(int k=0;k<3;k++){ lam_ee_pos[k]+=dlam[k]; x[k]+=c[k]; }
            }

            // EE orientation
            {
                double cq[4],dlam[3];
                solve_ee_orient(x.data()+3, imr[0], ee_quat,
                    lam_ee_ori, alpha_ee_orient, dt,
                    C_ee_ori_n, gamma_ee_orient, cq, dlam);
                for(int k=0;k<3;k++)  lam_ee_ori[k]+=dlam[k];
                for(int k=0;k<4;k++)  x[3+k]+=cq[k];
                normalise4(x.data()+3);
            }
        }

        // ── 5. Internal wrench recovery ───────────────────────────────────
        std::vector<double> F_int(3*N,0), T_int(3*N,0);
        double dt2=dt*dt;
        for(int i=1;i<N;i++){
            double R[9]; qrotm(&x[(i-1)*DOF+3], R);
            double* ls=&lam_str[3*(i-1)];
            double fs[3]; Rv(R, ls, fs);
            for(int k=0;k<3;k++) fs[k]/=dt2;
            for(int k=0;k<3;k++){ F_int[3*i+k]+=fs[k]; F_int[3*(i-1)+k]-=fs[k]; }
        }
        for(int i=1;i<N;i++){
            double* lb=&lam_ben[3*(i-1)];
            for(int k=0;k<3;k++){
                T_int[3*(i-1)+k]-=lb[k]/dt2;
                T_int[3*i+k]    +=lb[k]/dt2;
            }
        }

        // to local frames → CW
        auto CW_arr = py::array_t<double>(6*N);
        double* CW=CW_arr.mutable_data();
        for(int j=0;j<N;j++){
            double R[9]; qrotm(&x[j*DOF+3], R);
            double fl[3], tl[3];
            RTv(R, &F_int[3*j], fl);
            RTv(R, &T_int[3*j], tl);
            // column-major: CW[j], CW[N+j], ... 
            CW[j]     =fl[0]; CW[N+j]  =fl[1]; CW[2*N+j]=fl[2];
            CW[3*N+j] =tl[0]; CW[4*N+j]=tl[1]; CW[5*N+j]=tl[2];
        }

        // ── 6. EE wrench ──────────────────────────────────────────────────
        auto ew_arr=py::array_t<double>(6);
        double* ew=ew_arr.mutable_data();
        for(int k=0;k<3;k++){ ew[k]=lam_ee_pos[k]/dt2; ew[3+k]=lam_ee_ori[k]/dt2; }

        // ── 7. Velocity update ────────────────────────────────────────────
        for(int i=0;i<N;i++){
            int sx=i*DOF, sv=i*6;
            for(int k=0;k<3;k++) v[sv+k]=(x[sx+k]-xi(sx+k))/dt;

            double qn[4]={ x[sx+3],x[sx+4],x[sx+5],x[sx+6] };
            double qo[4]={ xi(sx+3),xi(sx+4),xi(sx+5),xi(sx+6) };
            if(dot4(qn,qo)<0){ for(int k=0;k<4;k++) qn[k]=-qn[k]; }
            double qoc[4]; qconj(qo,qoc);
            double qdiff[4]; qmul(qn,qoc,qdiff);
            for(int k=0;k<3;k++) v[sv+3+k]=(2.0/dt)*qdiff[k];
        }

        // ── Build output arrays ───────────────────────────────────────────
        auto x_out=py::array_t<double>(DOF*N);
        auto v_out=py::array_t<double>(6*N);
        double* xo=x_out.mutable_data(); double* vo=v_out.mutable_data();
        for(int i=0;i<DOF*N;i++) xo[i]=x[i];
        for(int i=0;i<6*N;i++)   vo[i]=v[i];

        return py::make_tuple(x_out, v_out, CW_arr, ew_arr);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Pybind11 module
// ════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(xpbd_core, m) {
    m.doc() = "C++ XPBD Cosserat-rod wire simulator";

    py::class_<WireSimulatorCPP>(m, "WireSimulatorCPP")
        .def(py::init<int,int,double,double,int,
                      carr,carr,carr,carr,
                      double,double,double,double,
                      carr,carr>(),
             py::arg("N"), py::arg("dof"), py::arg("L"), py::arg("dt"),
             py::arg("solver_iter"),
             py::arg("alpha_stretch"), py::arg("alpha_bend"),
             py::arg("beta_stretch"),  py::arg("beta_bend"),
             py::arg("alpha_ee_pos"),  py::arg("alpha_ee_orient"),
             py::arg("beta_ee_pos"),   py::arg("beta_ee_orient"),
             py::arg("m_num"),         py::arg("I_flat"))
        .def("estimate_wire_state",
             &WireSimulatorCPP::estimate_wire_state,
             py::arg("x_init"), py::arg("v_init"),
             py::arg("ee_pos"), py::arg("ee_quat"), py::arg("f_ext"),
             "One XPBD timestep.  Returns (x, v, CW, ee_wrench).");
}