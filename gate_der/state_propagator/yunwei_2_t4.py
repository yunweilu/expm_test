import numpy as np
from matplotlib.lines import Line2D
from expm_mul import expm_multiply
from scipy.sparse.linalg import eigs,norm
from scipy.special import factorial
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.sparse import csr_matrix,isspmatrix,bmat
import scipy as sci
def get_creation_operator(size,tp):
    return np.diag(np.sqrt(np.arange(1, size),dtype=tp), k=-1)
def get_annihilation_operator(size,tp):
    return np.diag(np.sqrt(np.arange(1, size),dtype=tp), k=1)

def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.
    if sci.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=0).flat)
    else:
        return np.linalg.norm(A, 1)

def gamma_fa(n):
    u = 1.11e-16
    return n*u/(1-n*u)
def beta(norm,m,n):
    beta=gamma_fa(m+1)
    r = 1
    for i in range(1,m):
        r=r*norm/i
        g = gamma_fa(i*(n+2)+m+2)
        beta = beta+g*r
    return beta,r

def taylor_term(i,norm,term):
    return term*norm/i
def error(norm_B,m,s,n,R_m):
    tr = R_m
    rd=beta(norm_B,m,n)[0]
    rd=np.power((1+rd+tr),s)-np.power((1+tr),s)
    tr=tr*s
    tr=tr*((1-np.power(tr,s))/(1-tr))
    return tr+rd
def weaker_error(beta,R_m,s):
    tr = R_m
    rd=beta
    rd = np.power((1 + rd + tr), s) - np.power((1 + tr), s)
    tr = tr * s
    tr = tr * ((1 - np.power(tr, s)) / (1 - tr))
    return tr+rd
def residue_norm(m,norm_B,term):
    R_m=term
    for i in range(m+2,1000):
        term=term*norm_B/i
        R_m=R_m+term
        if term<1e-15:
            break
    return R_m
def choose_ms(norm_A,d,tol):
    no_solution=True
    for i in range(1,int(np.floor(norm_A))):
        norm_B = norm_A / i
        l=int(np.ceil(norm_B))
        beta_factor,last_term=beta(norm_B,l,d)
        lower_bound = i*(beta_factor)
        if lower_bound>tol:
            continue
        tr_first_term=norm_B
        m_pass_lowbound=False
        for j in range(1,100):
            if j>l:
                beta_factor=beta_factor+gamma_fa(j*(d+2)+2)*last_term*norm_B/j
            if m_pass_lowbound == False:
                tr_first_term = tr_first_term * norm_B / (j + 1)
                if i * (tr_first_term + lower_bound) > tol:
                    continue
                else:
                    R_m = residue_norm(j, norm_B, tr_first_term)
                    m_pass_lowbound = True
            else:
                tr_first_term = tr_first_term * norm_B / (j + 1)
                R_m = R_m- tr_first_term
            if weaker_error(beta_factor,R_m,i)>tol:
                continue
            else:
                total_error=error(norm_B,j,i,d,R_m)
                if total_error<tol:
                    no_solution = False
                    s=i
                    m=j
                    break
        if no_solution == False:
            break
    if no_solution==False:
        return s,m
    if no_solution == True:
        raise ValueError("please lower the error tolerance ")
def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if sci.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    else:
        return np.linalg.norm(A, np.inf)
def max_row_number(sparse_matrix):
    row_indice=sparse_matrix.nonzero()[0]
    indice_count=1
    max_count=1
    length=len(row_indice)
    indice = row_indice[0]
    for i in range(1,length):
        if indice==row_indice[i]:
            indice_count=indice_count+1
        else:
            if indice_count>max_count:
                max_count=indice_count
            indice=row_indice[i]
            indice_count=1
    return max_count
def expm_yunwei(A, B,d, tol=None):
    """
    A helper function.
    """
    if tol is None:
        tol =1e-5
    # if sci.sparse.isspmatrix(A):
    #     d=max_row_number(A)
    # else:
    #     d=len(A)
    norm_A = _exact_inf_norm(A)
    s,m=choose_ms(norm_A,d,tol)
    F=B
    for i in range(int(s)):
        for j in range(m):
            coeff = s*(j+1)
            B =  A.dot(B)/coeff
            F = F + B
        B = F
    return F,s*m
def overnorm(A):
    if A.dtype==np.complex256:
        return _exact_inf_norm(A)
    else:
        return norm_two(A)
def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if sci.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    else:
        return np.linalg.norm(A, np.inf)
def norm_two(A):
    if sci.sparse.isspmatrix(A):
        A=csr_matrix(A).conjugate().transpose()
        return np.sqrt(abs(eigs(A=A.dot(A),k=1,which='LM',return_eigenvectors=False)[0]))
    else:
        return np.linalg.norm(A)
def norm_state(A):
    return np.linalg.norm(A)
def block_fre(A,E):
    if isspmatrix(A) is False:
        A = np.block([[A, E], [np.zeros_like(A), A]])
    else:
        A = bmat([[A, E], [None, A]]).tocsc()
    return A
def difference(A,B):
    return overnorm(A-B)/overnorm(A)
from qutip import qload
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.special import factorial
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix,identity
from fractions import Fraction
from scipy.sparse import csr_matrix,isspmatrix,bmat,kron
import scipy as sci
from scipy.sparse import csr_matrix,isspmatrix,bmat
def get_creation_operator(size,tp):
    return np.diag(np.sqrt(np.arange(1, size),dtype=tp), k=-1)
def get_annihilation_operator(size,tp):
    return np.diag(np.sqrt(np.arange(1, size),dtype=tp), k=1)
def get_norm(dim):
    tp=np.float64
    anorm=[]
    for i in range(len(dim)):
        N=7
        control=get_control(N,tp)
        control_pra=0.5*2*np.pi*np.ones(np.array(control).shape[0])
        H_control=control_H(control_pra,control)
        H=H_control+get_int(N,tp)
        anorm.append(overnorm(dim[i]*H))
    return anorm
def get_norm_der(dim):
    tp=np.float64
    anorm=[]
    for i in range(len(dim)):
        N=7
        control=get_control(N,tp)
        control_pra=0.5*2*np.pi*np.ones(np.array(control).shape[0])
        H_control=control_H(control_pra,control)
        H=H_control+get_int(N,tp)
        vec=1/np.sqrt(2*2**N)*np.identity(2*2**N)
        A=block_fre(H,-1j*(control[0]))
        anorm.append(overnorm(dim[i]*A))
    return anorm
def get_auxiliary(N,tp,t):
    control=get_control(N,tp)
    H,vec = get_H(N,tp)
    vec=1/np.sqrt(2**N)*np.ones(2*2**N)
    vec[0:2**N]=0
    A=block_fre(t*H,-1j*(control[3]))
    return A,vec
def get_int(N,tp):
    sigmap=get_creation_operator(2,tp)
    sigmam = get_annihilation_operator(2,tp)
    sigmaz=sigmap.dot(sigmam)
    H0=0
    SIGMAZ=kron(sigmaz,sigmaz)
    H0=H0+kron(SIGMAZ,identity(2**(N-2)))+kron(identity(2**(N-2)),SIGMAZ)
    for i in range(1,N-2):
        H0=H0+kron(kron(identity(2**i),SIGMAZ),identity(2 ** (N - 2 - i)))
    return H0
def get_control(N,tp):
    sigmap=get_creation_operator(2,tp)
    sigmam = get_annihilation_operator(2,tp)
    sigmap=sigmap
    sigmam=sigmam
    sigmax=sigmap+sigmam
    sigmay=-1j*sigmap+1j*sigmam
    control=[]
    if N==1:
        control.append(kron(sigmax, identity(2 ** (N - 1))))
        control.append(kron(sigmay, identity(2 ** (N - 1))))
        return control
    else:
        a=identity(2**(N-1))
        control.append(kron(sigmax,a,format="csc"))
        control.append(kron(sigmay, identity(2 ** (N - 1)),format="csc"))
        for i in range(1,N-1):
            control.append(kron(kron(identity(2**i),sigmax), identity(2 ** (N - 1-i)),format="csc"))
            control.append(kron(kron(identity(2 ** i), sigmay), identity(2 ** (N - 1 - i)),format="csc"))
        control.append(kron(identity(2**(N-1)),sigmax,format="csc"))
        control.append(kron(identity(2**(N-1)),sigmay,format="csc"))
    return control
def control_H(control,H_control):
    H=0
    for i in range(len(control)):
        H=H+control[i]*H_control[i]
    return H
def get_H(N,tp):
    control=get_control(N,tp)
    control_pra=0.5*2*np.pi*np.ones(np.array(control).shape[0])
    H_control=control_H(control_pra,control)
    H=-1j*(H_control+get_int(N,tp)*0.1*2*np.pi)
    vec=1/np.sqrt(2**N)*np.ones(2**N)
    return H,vec

theta_m=np.array([[8.70950435e-01, 2.48853043e+00, 4.02062662e+00, 5.51075280e+00,
        6.97775212e+00, 8.43027209e+00, 9.87290220e+00, 1.13089234e+01,
        1.27393366e+01, 1.41653119e+01, 1.55887688e+01],
       [1.79283319e-01, 1.30547755e+00, 2.68990434e+00, 4.11674468e+00,
        5.55078449e+00, 6.98409736e+00, 8.41474411e+00, 9.84240944e+00,
        1.12672525e+01, 1.26895571e+01, 1.41096178e+01],
       [0.13,1, 2.2, 3.6,
        4.9, 6.3, 7.7, 9.1,
        11, 12, 1.3],
       [5.87845778e-03, 3.11153531e-01, 1.11496247e+00, 2.19235380e+00,
        3.40348589e+00, 4.68492721e+00, 6.00642014e+00, 7.35228229e+00,
        8.71376145e+00, 1.00856663e+01, 1.14647607e+01],
        [0.003307471010225917,0.24259255239783006,0.9560279579613657,1.9645322216064995,3.1267463661459876,4.372820117778894,5.667919829026178,6.993478543263042,8.338922480962493,9.697877908033124,11.066316441049558 ],
       [2.3e-03, 0.14, 0.64, 1.4,
        2.4, 3.5, 4.7, 6,
        7.2, 8.5, 9.9]])
theta_mm=[]
for j,_theta in enumerate(theta_m):
    a={5:0,10:0,15:0,20:0,25:0,30:0,35:0,40:0,45:0,50:0,55:0}
    for i, theta in enumerate(_theta):
        a[5*(i+1)] = theta
    theta_mm.append(a)
tol = 1e-2
dim=7
t=4
H,vec=get_auxiliary(dim,np.float64,1)
for i in range(500):
    a,x1=expm_yunwei(t*H, vec,9,tol)