// Distributed under MIT licence. See https://github.com/aniabrown/QuEST_GPU/blob/master/LICENCE.txt for details

/** @file
 * Internal and API functions which are hardware-agnostic.
 * These must never call a front-end function in QuEST.c, which would lead to 
 * duplication of e.g. QASM logging and validation. Note that though many of
 * these functions are prefixed with statevec_, they will be called multiple times
 * to effect their equivalent operation on density matrices, so the passed Qureg
 * can be assumed a statevector. Functions prefixed with densmatr_ may still
 * explicitly call statevec_ functions, but will need to manually apply the
 * conjugate qubit-shifted operations to satisfy the Choi–Jamiolkowski isomorphism
 */

# include "QuEST.h"
# include "QuEST_internal.h"
# include "QuEST_precision.h"
# include "QuEST_validation.h"
# include "mt19937ar.h"

# define _BSD_SOURCE
# include <unistd.h>
# include <sys/types.h> 
# include <sys/time.h>
# include <sys/param.h>
# include <stdio.h>
# include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

/* builds a bit-string where 1 indicates a qubit is present in this list */
long long int getQubitBitMask(int* qubits, const int numQubits) {
    
    long long int mask=0; 
    for (int i=0; i<numQubits; i++)
        mask = mask | (1LL << qubits[i]);
        
    return mask;
}

/* builds a bit-string where 1 indicates control qubits conditioned on 0 ('flipped') */
long long int getControlFlipMask(int* controlQubits, int* controlState, const int numControlQubits) {
    
    long long int mask=0;
    for (int i=0; i<numControlQubits; i++)
        if (controlState[i] == 0)
            mask = mask | (1LL << controlQubits[i]);
            
    return mask;
}

/* modifies ind1 and ind2 so that ind2 >= ind1 */
void ensureIndsIncrease(int* ind1, int* ind2) {
    if (*ind1 > *ind2) {
        int copy = *ind1;
        *ind1 = *ind2;
        *ind2 = copy;
    }
}

/* returns the Euclidean norm of a real vector */
qreal getVectorMagnitude(Vector vec) {
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

Vector getUnitVector(Vector vec) {
    
    qreal mag = getVectorMagnitude(vec);
    Vector unitVec = (Vector) {.x=vec.x/mag, .y=vec.y/mag, .z=vec.z/mag};
    return unitVec;
}

Complex getConjugateScalar(Complex scalar) {
    
    Complex conjScalar;
    conjScalar.real =   scalar.real;
    conjScalar.imag = - scalar.imag;
    return conjScalar;
}

ComplexMatrix2 getConjugateMatrix(ComplexMatrix2 matrix) {
    
    ComplexMatrix2 conjMatrix;
    conjMatrix.r0c0 = getConjugateScalar(matrix.r0c0);
    conjMatrix.r0c1 = getConjugateScalar(matrix.r0c1);
    conjMatrix.r1c0 = getConjugateScalar(matrix.r1c0);
    conjMatrix.r1c1 = getConjugateScalar(matrix.r1c1);
    return conjMatrix;
}

void getComplexPairFromRotation(qreal angle, Vector axis, Complex* alpha, Complex* beta) {
    
    Vector unitAxis = getUnitVector(axis);
    alpha->real =   cos(angle/2.0);
    alpha->imag = - sin(angle/2.0)*unitAxis.z;  
    beta->real  =   sin(angle/2.0)*unitAxis.y;
    beta->imag  = - sin(angle/2.0)*unitAxis.x;
}

/** maps U(alpha, beta) to Rz(rz2) Ry(ry) Rz(rz1) */
void getZYZRotAnglesFromComplexPair(Complex alpha, Complex beta, qreal* rz2, qreal* ry, qreal* rz1) {
    
    qreal alphaMag = sqrt(alpha.real*alpha.real + alpha.imag*alpha.imag);
    *ry = 2.0 * acos(alphaMag);
    
    qreal alphaPhase = atan2(alpha.imag, alpha.real);
    qreal betaPhase  = atan2(beta.imag,  beta.real);
    *rz2 = - alphaPhase + betaPhase;
    *rz1 = - alphaPhase - betaPhase;
}

/** maps U(r0c0, r0c1, r1c0, r1c1) to exp(i globalPhase) U(alpha, beta) */
void getComplexPairAndPhaseFromUnitary(ComplexMatrix2 u, Complex* alpha, Complex* beta, qreal* globalPhase) {
    
    qreal r0c0Phase = atan2(u.r0c0.imag, u.r0c0.real);
    qreal r1c1Phase = atan2(u.r1c1.imag, u.r1c1.real);
    *globalPhase = (r0c0Phase + r1c1Phase)/2.0;
    
    qreal cosPhase = cos(*globalPhase);
    qreal sinPhase = sin(*globalPhase);
    alpha->real = u.r0c0.real*cosPhase + u.r0c0.imag*sinPhase;
    alpha->imag = u.r0c0.imag*cosPhase - u.r0c0.real*sinPhase;
    beta->real = u.r1c0.real*cosPhase + u.r1c0.imag*sinPhase;
    beta->imag = u.r1c0.imag*cosPhase - u.r1c0.real*sinPhase;
}

void shiftIndices(int* indices, int numIndices, int shift) {
    for (int j=0; j < numIndices; j++)
        indices[j] += shift;
}

int generateMeasurementOutcome(qreal zeroProb, qreal *outcomeProb) {
    
    // randomly choose outcome
    int outcome;
    if (zeroProb < REAL_EPS) 
        outcome = 1;
    else if (1-zeroProb < REAL_EPS) 
        outcome = 0;
    else
        outcome = (genrand_real1() > zeroProb);
    
    // set probability of outcome
    if (outcome == 0)
        *outcomeProb = zeroProb;
    else
        *outcomeProb = 1 - zeroProb;
    
    return outcome;
}

unsigned long int hashString(char *str){
    unsigned long int hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;    
}

void getQuESTDefaultSeedKey(unsigned long int *key){
    // init MT random number generator with three keys -- time, pid and a hash of hostname 
    // for the MPI version, it is ok that all procs will get the same seed as random numbers will only be 
    // used by the master process

    struct timeval  tv;
    gettimeofday(&tv, NULL);

    double time_in_mill =
        (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ; // convert tv_sec & tv_usec to millisecond

    unsigned long int pid = getpid();
    unsigned long int msecs = (unsigned long int) time_in_mill;
    char hostName[MAXHOSTNAMELEN+1];
    gethostname(hostName, sizeof(hostName));
    unsigned long int hostNameInt = hashString(hostName);

    key[0] = msecs; key[1] = pid; key[2] = hostNameInt;
}

/** 
 * numSeeds <= 64
 */
void seedQuEST(unsigned long int *seedArray, int numSeeds){
    // init MT random number generator with user defined list of seeds
    // for the MPI version, it is ok that all procs will get the same seed as random numbers will only be 
    // used by the master process
    init_by_array(seedArray, numSeeds); 
}

void reportState(Qureg qureg){
    FILE *state;
    char filename[100];
    long long int index;
    sprintf(filename, "state_rank_%d.csv", qureg.chunkId);
    state = fopen(filename, "w");
    if (qureg.chunkId==0) fprintf(state, "real, imag\n");

    for(index=0; index<qureg.numAmpsPerChunk; index++){
        # if QuEST_PREC==1 || QuEST_PREC==2
        fprintf(state, "%.12f, %.12f\n", qureg.stateVec.real[index], qureg.stateVec.imag[index]);
        # elif QuEST_PREC == 4
        fprintf(state, "%.12Lf, %.12Lf\n", qureg.stateVec.real[index], qureg.stateVec.imag[index]);
        #endif
    }
    fclose(state);
}

void reportQuregParams(Qureg qureg){
    long long int numAmps = 1L << qureg.numQubitsInStateVec;
    long long int numAmpsPerRank = numAmps/qureg.numChunks;
    if (qureg.chunkId==0){
        printf("QUBITS:\n");
        printf("Number of qubits is %d.\n", qureg.numQubitsInStateVec);
        printf("Number of amps is %lld.\n", numAmps);
        printf("Number of amps per rank is %lld.\n", numAmpsPerRank);
    }
}

qreal statevec_getProbAmp(Qureg qureg, long long int index){
    qreal real = statevec_getRealAmp(qureg, index);
    qreal imag = statevec_getImagAmp(qureg, index);
    return real*real + imag*imag;
}

void statevec_phaseShift(Qureg qureg, const int targetQubit, qreal angle) {
    Complex term; 
    term.real = cos(angle); 
    term.imag = sin(angle);
    statevec_phaseShiftByTerm(qureg, targetQubit, term);
}

void statevec_pauliZ(Qureg qureg, const int targetQubit) {
    Complex term; 
    term.real = -1;
    term.imag =  0;
    statevec_phaseShiftByTerm(qureg, targetQubit, term);
}

void statevec_sGate(Qureg qureg, const int targetQubit) {
    Complex term; 
    term.real = 0;
    term.imag = 1;
    statevec_phaseShiftByTerm(qureg, targetQubit, term);
} 

void statevec_tGate(Qureg qureg, const int targetQubit) {
    Complex term; 
    term.real = 1/sqrt(2);
    term.imag = 1/sqrt(2);
    statevec_phaseShiftByTerm(qureg, targetQubit, term);
}

void statevec_sGateConj(Qureg qureg, const int targetQubit) {
    Complex term; 
    term.real =  0;
    term.imag = -1;
    statevec_phaseShiftByTerm(qureg, targetQubit, term);
} 

void statevec_tGateConj(Qureg qureg, const int targetQubit) {
    Complex term; 
    term.real =  1/sqrt(2);
    term.imag = -1/sqrt(2);
    statevec_phaseShiftByTerm(qureg, targetQubit, term);
}

void statevec_rotateX(Qureg qureg, const int rotQubit, qreal angle){

    Vector unitAxis = {1, 0, 0};
    statevec_rotateAroundAxis(qureg, rotQubit, angle, unitAxis);
}

void statevec_rotateY(Qureg qureg, const int rotQubit, qreal angle){

    Vector unitAxis = {0, 1, 0};
    statevec_rotateAroundAxis(qureg, rotQubit, angle, unitAxis);
}

void statevec_rotateZ(Qureg qureg, const int rotQubit, qreal angle){

    Vector unitAxis = {0, 0, 1};
    statevec_rotateAroundAxis(qureg, rotQubit, angle, unitAxis);
}

void statevec_rotateAroundAxis(Qureg qureg, const int rotQubit, qreal angle, Vector axis){

    Complex alpha, beta;
    getComplexPairFromRotation(angle, axis, &alpha, &beta);
    statevec_compactUnitary(qureg, rotQubit, alpha, beta);
}

void statevec_rotateAroundAxisConj(Qureg qureg, const int rotQubit, qreal angle, Vector axis){

    Complex alpha, beta;
    getComplexPairFromRotation(angle, axis, &alpha, &beta);
    alpha.imag *= -1; 
    beta.imag *= -1;
    statevec_compactUnitary(qureg, rotQubit, alpha, beta);
}

void statevec_controlledRotateAroundAxis(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle, Vector axis){

    Complex alpha, beta;
    getComplexPairFromRotation(angle, axis, &alpha, &beta);
    statevec_controlledCompactUnitary(qureg, controlQubit, targetQubit, alpha, beta);
}

void statevec_controlledRotateAroundAxisConj(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle, Vector axis){

    Complex alpha, beta;
    getComplexPairFromRotation(angle, axis, &alpha, &beta);
    alpha.imag *= -1; 
    beta.imag *= -1;
    statevec_controlledCompactUnitary(qureg, controlQubit, targetQubit, alpha, beta);
}

void statevec_controlledRotateX(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle){

    Vector unitAxis = {1, 0, 0};
    statevec_controlledRotateAroundAxis(qureg, controlQubit, targetQubit, angle, unitAxis);
}

void statevec_controlledRotateY(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle){

    Vector unitAxis = {0, 1, 0};
    statevec_controlledRotateAroundAxis(qureg, controlQubit, targetQubit, angle, unitAxis);
}

void statevec_controlledRotateZ(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle){

    Vector unitAxis = {0, 0, 1};
    statevec_controlledRotateAroundAxis(qureg, controlQubit, targetQubit, angle, unitAxis);
}

int statevec_measureWithStats(Qureg qureg, int measureQubit, qreal *outcomeProb) {
    
    qreal zeroProb = statevec_calcProbOfOutcome(qureg, measureQubit, 0);
    int outcome = generateMeasurementOutcome(zeroProb, outcomeProb);
    statevec_collapseToKnownProbOutcome(qureg, measureQubit, outcome, *outcomeProb);
    return outcome;
}

int densmatr_measureWithStats(Qureg qureg, int measureQubit, qreal *outcomeProb) {
    
    qreal zeroProb = densmatr_calcProbOfOutcome(qureg, measureQubit, 0);
    int outcome = generateMeasurementOutcome(zeroProb, outcomeProb);
    densmatr_collapseToKnownProbOutcome(qureg, measureQubit, outcome, *outcomeProb);
    return outcome;
}

qreal statevec_calcFidelity(Qureg qureg, Qureg pureState) {
    
    Complex innerProd = statevec_calcInnerProduct(qureg, pureState);
    qreal innerProdMag = innerProd.real*innerProd.real + innerProd.imag*innerProd.imag;
    return innerProdMag;
}

void statevec_swapGate(Qureg qureg, int qb1, int qb2) {

    statevec_controlledNot(qureg, qb1, qb2);
    statevec_controlledNot(qureg, qb2, qb1);
    statevec_controlledNot(qureg, qb1, qb2);
}

void statevec_sqrtSwapGate(Qureg qureg, int qb1, int qb2) {
    
    ComplexMatrix2 u;
    u.r0c0.real = .5; u.r0c0.imag = .5;
    u.r0c1.real = .5; u.r0c1.imag =-.5;
    u.r1c0.real = .5; u.r1c0.imag =-.5;
    u.r1c1.real = .5; u.r1c1.imag = .5;
    
    statevec_controlledNot(qureg, qb1, qb2);
    statevec_controlledUnitary(qureg, qb2, qb1, u);
    statevec_controlledNot(qureg, qb1, qb2);
}

void statevec_sqrtSwapGateConj(Qureg qureg, int qb1, int qb2) {
    
    ComplexMatrix2 u;
    u.r0c0.real = .5; u.r0c0.imag =-.5;
    u.r0c1.real = .5; u.r0c1.imag = .5;
    u.r1c0.real = .5; u.r1c0.imag = .5;
    u.r1c1.real = .5; u.r1c1.imag =-.5;
    
    statevec_controlledNot(qureg, qb1, qb2);
    statevec_controlledUnitary(qureg, qb2, qb1, u);
    statevec_controlledNot(qureg, qb1, qb2);
}

void densmatr_oneQubitPauliError(Qureg qureg, int qubit, qreal pX, qreal pY, qreal pZ) {
    
    // The accepted pX, pY, pZ do NOT need to be pre-scaled.
    // The passed probabilities below are modified so that the final state produced is
    // q + px X q X + py Y q Y + pz Z q Z
    // The *2 are due to oneQubitDephase accepting 'dephaseLevel', not a probability
    // The double statevec_compactUnitary calls are needed (with complex conjugates)
    // since we're operating on a density matrix, not a statevector, and this function
    // won't be called twice from the front-end
    
    Complex alpha, beta;
    qreal fac = 1/sqrt(2);
    int numQb = qureg.numQubitsRepresented;
    
    // add Z error
    densmatr_oneQubitDephase(qureg, qubit, 2 * pZ/(1-pX-pY));
    
    // rotate basis via Rx(pi/2)
    alpha.real = fac; alpha.imag = 0;
    beta.real = 0;    beta.imag = -fac;
    statevec_compactUnitary(qureg, qubit, alpha, beta);
    alpha.imag *= -1; beta.imag *= -1;
    statevec_compactUnitary(qureg, qubit + numQb, alpha, beta);
    
    // add Z -> Y Rx(pi/2) error 
    densmatr_oneQubitDephase(qureg, qubit, 2 * pY/(1-pX));
    
    // rotate basis by Rx(-pi/2) then Ry(pi/2) 
    alpha.real = .5; alpha.imag = -.5;
    beta.real = .5;  beta.imag = .5;
    statevec_compactUnitary(qureg, qubit, alpha, beta);
    alpha.imag *= -1; beta.imag *= -1;
    statevec_compactUnitary(qureg, qubit + numQb, alpha, beta);
    
    // add Z -> X Ry(pi/2) error
    densmatr_oneQubitDephase(qureg, qubit, 2 * pX);
    
    // restore basis
    alpha.real = fac; alpha.imag = 0;
    beta.real = -fac; beta.imag = 0;
    statevec_compactUnitary(qureg, qubit, alpha, beta);
    alpha.imag *= -1; beta.imag *= -1;
    statevec_compactUnitary(qureg, qubit + numQb, alpha, beta);
}

/** applyConj=1 will apply conjugate operation, else applyConj=0 */
void statevec_multiRotatePauli(
    Qureg qureg, int* targetQubits, int* targetPaulis, int numTargets, qreal angle,
    int applyConj
) {
    qreal fac = 1/sqrt(2);
    Complex uRxAlpha = {.real = fac, .imag = 0}; // Rx(pi/2)* rotates Z -> Y
    Complex uRxBeta = {.real = 0, .imag = (applyConj)? fac : -fac};
    Complex uRyAlpha = {.real = fac, .imag = 0}; // Ry(pi/2) rotates Z -> X
    Complex uRyBeta = {.real = fac, .imag = 0};
    
    // mask may be modified to remove superfluous Identity ops
    long long int mask = getQubitBitMask(targetQubits, numTargets);
    
    // rotate basis so that exp(Z) will effect exp(Y) and exp(X)
    for (int t=0; t < numTargets; t++) {
        if (targetPaulis[t] == 0)
            mask -= 1LL << targetPaulis[t]; // remove target from mask
        if (targetPaulis[t] == 1)
            statevec_compactUnitary(qureg, targetQubits[t], uRyAlpha, uRyBeta);
        if (targetPaulis[t] == 2)
            statevec_compactUnitary(qureg, targetQubits[t], uRxAlpha, uRxBeta);
        // (targetPaulis[t] == 3) is Z basis
    }
    
    statevec_multiRotateZ(qureg, mask, (applyConj)? -angle : angle);
    
    // undo X and Y basis rotations
    uRxBeta.imag *= -1;
    uRyBeta.real *= -1;
    for (int t=0; t < numTargets; t++) {
        if (targetPaulis[t] == 1)
            statevec_compactUnitary(qureg, targetQubits[t], uRyAlpha, uRyBeta);
        if (targetPaulis[t] == 2)
            statevec_compactUnitary(qureg, targetQubits[t], uRxAlpha, uRxBeta);
    }
}

Complex getScalarProduct(Complex a, Complex b) {
    return (Complex) {
        .real = a.real*b.real - a.imag*b.imag,
        .imag = a.real*b.imag + b.real*a.imag};
}

Complex getScalarFactor(qreal fac, Complex a) {
    return getScalarProduct(a, (Complex) {.real=fac, .imag=0});
}

Complex getScalarSum(Complex a, Complex b) {
    return (Complex) {.real = a.real + b.real, .imag = a.imag + b.imag};
}

/* returns the principal square root of a complex scalar */
Complex getScalarSqrt(Complex a) {
    qreal x = a.real; 
    qreal y = a.imag;
    Complex s;
    s.real = sqrt(2.0)/2.0 * sqrt( x + sqrt(x*x + y*y));
    s.imag = sqrt(2.0)/2.0 * sqrt(-x + sqrt(x*x + y*y)) * ((y>0)? 1:-1);
    return s;
}

/* returns a/b */
Complex getScalarQuotient(Complex a, Complex b) {
    qreal denom = b.real*b.real + b.imag*b.imag;
    Complex quot;
    quot.real = (a.real*b.real + a.imag*b.imag)/denom;
    quot.imag = (a.imag*b.real - a.real*b.imag)/denom;
    return quot;
}

ComplexMatrix2 getMatrixProduct(ComplexMatrix2 a, ComplexMatrix2 b) {
    ComplexMatrix2 prod;
    prod.r0c0 = getScalarSum(getScalarProduct(a.r0c0, b.r0c0), getScalarProduct(a.r0c1, b.r1c0));
    prod.r0c1 = getScalarSum(getScalarProduct(a.r0c0, b.r0c1), getScalarProduct(a.r0c1, b.r1c1));
    prod.r1c0 = getScalarSum(getScalarProduct(a.r1c0, b.r0c0), getScalarProduct(a.r1c1, b.r1c0));
    prod.r1c1 = getScalarSum(getScalarProduct(a.r1c0, b.r0c1), getScalarProduct(a.r1c1, b.r1c1));
    return prod;
}

ComplexMatrix2 getDaggerMatrix(ComplexMatrix2 matrix) {
    
    ComplexMatrix2 conjMatrix;
    conjMatrix.r0c0 = getConjugateScalar(matrix.r0c0);
    conjMatrix.r0c1 = getConjugateScalar(matrix.r1c0); // swapped 
    conjMatrix.r1c0 = getConjugateScalar(matrix.r0c1); // swapped
    conjMatrix.r1c1 = getConjugateScalar(matrix.r1c1);
    return conjMatrix;
}

ComplexMatrix2 getSubMatrix(ComplexMatrix4 u, int r, int c) {
    ComplexMatrix2 sub;
    if (r==0 && c==0) {
        sub.r0c0 = u.r0c0; sub.r0c1 = u.r0c1;
        sub.r1c0 = u.r1c0; sub.r1c1 = u.r1c1;
    }
    if (r==0 && c==1) {
        sub.r0c0 = u.r0c2; sub.r0c1 = u.r0c3;
        sub.r1c0 = u.r1c2; sub.r1c1 = u.r1c3;
    }
    if (r==1 && c==0) {
        sub.r0c0 = u.r2c0; sub.r0c1 = u.r2c1;
        sub.r1c0 = u.r3c0; sub.r1c1 = u.r3c1;
    }
    if (r==1 && c==1) {
        sub.r0c0 = u.r2c2; sub.r0c1 = u.r2c3;
        sub.r1c0 = u.r3c2; sub.r1c1 = u.r3c3;
    }
    return sub;
}

qreal getScalarMagSquared(Complex a) {
    return a.real*a.real + a.imag*a.imag;
}

void normaliseColumns(ComplexMatrix2* vecs) {
    qreal lnorm = 1.0/sqrt(getScalarMagSquared(vecs->r0c0) + getScalarMagSquared(vecs->r1c0));
    vecs->r0c0 = getScalarFactor(lnorm, vecs->r0c0);
    vecs->r1c0 = getScalarFactor(lnorm, vecs->r1c0);
    qreal rnorm = 1.0/sqrt(getScalarMagSquared(vecs->r0c1) + getScalarMagSquared(vecs->r1c1));
    vecs->r0c1 = getScalarFactor(rnorm, vecs->r0c1);
    vecs->r1c1 = getScalarFactor(rnorm, vecs->r1c1);
}

// DEBUG 
void printComp(Complex a) {
    printf("%g + i(%g)", a.real, a.imag);
}
void printMatrix(ComplexMatrix2 u) {
    printf("r0c0: "); printComp(u.r0c0); printf("\n");
    printf("r0c1: "); printComp(u.r0c1); printf("\n");
    printf("r1c0: "); printComp(u.r1c0); printf("\n");
    printf("r1c1: "); printComp(u.r1c1); printf("\n");
}

/* performs an eigendecomposition of u, populating vecs,vals such that 
 * the columns of vecs are the eigenvectors, and val is diagonal with the 
 * eigenvalues of u, in descending order */
void eigenDecompose(ComplexMatrix2 u, ComplexMatrix2* vecs, ComplexMatrix2* vals) {
    Complex a, b, c, d;
    a = u.r0c0; b = u.r0c1;
    c = u.r1c0; d = u.r1c1;
    
    // g = sqrt(a^2 + 4 b c - 2 a d + d^2)
    Complex g;
    g = getScalarProduct(a, a);
    g = getScalarSum(g, getScalarFactor(4, getScalarProduct(b, c)));
    g = getScalarSum(g, getScalarFactor(-2, getScalarProduct(a, d)));
    g = getScalarSum(g, getScalarProduct(d, d));
    g = getScalarSqrt(g);
    
    // set eigenvals to (a + d +- g)/2
    vals->r0c0 = getScalarFactor(.5, getScalarSum(
        getScalarSum(a,d), getScalarFactor(+1,g)));
    vals->r1c1 = getScalarFactor(.5, getScalarSum(
        getScalarSum(a,d), getScalarFactor(-1,g)));
    vals->r0c1 = (Complex) {.real=0, .imag=0};
    vals->r1c0 = (Complex) {.real=0, .imag=0};
    
    // set eigenvectors to {(a - d +- g)/2c, 1}
    vecs->r0c0 = getScalarQuotient(
        getScalarSum(
            getScalarSum(a, getScalarFactor(-1,d)),
            getScalarFactor(+1,g)),
        getScalarFactor(2,c));
    vecs->r0c1 = getScalarQuotient(
        getScalarSum(
            getScalarSum(a, getScalarFactor(-1,d)),
            getScalarFactor(-1,g)),
        getScalarFactor(2,c));
    vecs->r1c0 = (Complex) {.real=1, .imag=0};
    vecs->r1c1 = (Complex) {.real=1, .imag=0};
    
    // normalise eigenvectors 
    normaliseColumns(vecs);
}

/* performs singular value decomposition on u, populating l,d,r such that 
 * the columns of l are the eigenvectors of u u^dagger, the columns of r^dagger are the 
 * eigenvectors of u^dagger u, and d is a diagonal matrix of the "singular values"
 * of u which are equal to the square root of the eigenvalues of u^dagger u 
 * (equal to those of u u^dagger). Resultantly, u = l d r. This differs from the 
 * canonical form of SVD by the dagger'ing (conjugate transposing) of the r matrix. 
 * The returned l and r vectors are always unitary, even if the given u is not.
 */
void singularValueDecompose(ComplexMatrix2 u, ComplexMatrix2* l, ComplexMatrix2* d, ComplexMatrix2* r) {

    // get l by eigvecs(u u^dagger)
    ComplexMatrix2 uuT = getMatrixProduct(u, getDaggerMatrix(u));
    eigenDecompose(uuT, l, d);
    
    // get r by transforming lvecs
    *r = getMatrixProduct(getDaggerMatrix(u), *l);
    normaliseColumns(r);
    *r = getDaggerMatrix(*r);    

    // fix diagonals to be sqrt(eigvals(u u^dagger)) which should already be real
    d->r0c0.real = sqrt(d->r0c0.real);
    d->r1c1.real = sqrt(d->r1c1.real);
}

/* performs the cosine-sine decomposition on u, populating l0,l1,d0,d1,r0,r1 such that
 * u = {{u00, u01}, {u10,u11}} = {{l0,0},{0,l1}} {{d0, d1},{-d1,d0}} {{r0,0},{0,r1}}
 * where l0,l1,r0,r1 are unitary and d0 and d1 are diagonal. u does not need to be unitary.
 */
void cosineSineDecompose(
    ComplexMatrix4 u, 
    ComplexMatrix2* l0, ComplexMatrix2* l1,
    ComplexMatrix2* d0,  ComplexMatrix2* d1,
    ComplexMatrix2* r0, ComplexMatrix2* r1
) {
    // compute l0,d0,r0 via u00 = l0 d0 r0 (via SVD)
    singularValueDecompose(getSubMatrix(u, 0,0), l0, d0, r0);
    
    // compute d1,r1 via u01 = l0 d1 r1 (via SVD) 
    singularValueDecompose(getSubMatrix(u, 0,1), l0, d1, r1);
    
    // compute l1 via u11 = l1 d0 r1 (via SVD)
    //singularValueDecompose(getSubMatrix(u, 1,1), l1, d0, r1);
    
    // compute l1 via l1 = u11 dagger(r1) inv(d0)
    ComplexMatrix2 d0inv = {0};
    d0inv.r0c0.real = 1.0 / d0->r0c0.real; 
    d0inv.r1c1.real = 1.0 / d0->r1c1.real;
    *l1 = getMatrixProduct(
        getSubMatrix(u, 1,1),
        getMatrixProduct(
            getDaggerMatrix(*r1),
            d0inv));
}

void statevec_twoQubitUnitary(Qureg qureg, const int qb1, const int qb2, ComplexMatrix4 u) {
    
    // cosine-sine decompose u into {{l0,0},{0,l1}} {{d0, d1},{-d1,d0}} {{r0,0},{0,r1}}
    ComplexMatrix2 *l0, *l1, *d0, *d1, *r0, *r1;
    cosineSineDecompose(u, l0, l1, d0, d1, r0, r1);
    
    // @TODO
}








ComplexMatrix4 get4x4MatrixProduct(ComplexMatrix4 a, ComplexMatrix4 b) {
    ComplexMatrix4 combinedMatrix;
    combinedMatrix.r0c0 = getScalarSum(getScalarSum(getScalarProduct(a.r0c0, b.r0c0), getScalarProduct(a.r0c1, b.r1c0)), getScalarSum(getScalarProduct(a.r0c2, b.r2c0), getScalarProduct(a.r0c3, b.r3c0)));
    combinedMatrix.r0c1 = getScalarSum(getScalarSum(getScalarProduct(a.r0c0, b.r0c1), getScalarProduct(a.r0c1, b.r1c1)), getScalarSum(getScalarProduct(a.r0c2, b.r2c1), getScalarProduct(a.r0c3, b.r3c1)));
    combinedMatrix.r0c2 = getScalarSum(getScalarSum(getScalarProduct(a.r0c0, b.r0c2), getScalarProduct(a.r0c1, b.r1c2)), getScalarSum(getScalarProduct(a.r0c2, b.r2c2), getScalarProduct(a.r0c3, b.r3c2)));
    combinedMatrix.r0c3 = getScalarSum(getScalarSum(getScalarProduct(a.r0c0, b.r0c3), getScalarProduct(a.r0c1, b.r1c3)), getScalarSum(getScalarProduct(a.r0c2, b.r2c3), getScalarProduct(a.r0c3, b.r3c3)));
    combinedMatrix.r1c0 = getScalarSum(getScalarSum(getScalarProduct(a.r1c0, b.r0c0), getScalarProduct(a.r1c1, b.r1c0)), getScalarSum(getScalarProduct(a.r1c2, b.r2c0), getScalarProduct(a.r1c3, b.r3c0)));
    combinedMatrix.r1c1 = getScalarSum(getScalarSum(getScalarProduct(a.r1c0, b.r0c1), getScalarProduct(a.r1c1, b.r1c1)), getScalarSum(getScalarProduct(a.r1c2, b.r2c1), getScalarProduct(a.r1c3, b.r3c1)));
    combinedMatrix.r1c2 = getScalarSum(getScalarSum(getScalarProduct(a.r1c0, b.r0c2), getScalarProduct(a.r1c1, b.r1c2)), getScalarSum(getScalarProduct(a.r1c2, b.r2c2), getScalarProduct(a.r1c3, b.r3c2)));
    combinedMatrix.r1c3 = getScalarSum(getScalarSum(getScalarProduct(a.r1c0, b.r0c3), getScalarProduct(a.r1c1, b.r1c3)), getScalarSum(getScalarProduct(a.r1c2, b.r2c3), getScalarProduct(a.r1c3, b.r3c3)));
    combinedMatrix.r2c0 = getScalarSum(getScalarSum(getScalarProduct(a.r2c0, b.r0c0), getScalarProduct(a.r2c1, b.r1c0)), getScalarSum(getScalarProduct(a.r2c2, b.r2c0), getScalarProduct(a.r2c3, b.r3c0)));
    combinedMatrix.r2c1 = getScalarSum(getScalarSum(getScalarProduct(a.r2c0, b.r0c1), getScalarProduct(a.r2c1, b.r1c1)), getScalarSum(getScalarProduct(a.r2c2, b.r2c1), getScalarProduct(a.r2c3, b.r3c1)));
    combinedMatrix.r2c2 = getScalarSum(getScalarSum(getScalarProduct(a.r2c0, b.r0c2), getScalarProduct(a.r2c1, b.r1c2)), getScalarSum(getScalarProduct(a.r2c2, b.r2c2), getScalarProduct(a.r2c3, b.r3c2)));
    combinedMatrix.r2c3 = getScalarSum(getScalarSum(getScalarProduct(a.r2c0, b.r0c3), getScalarProduct(a.r2c1, b.r1c3)), getScalarSum(getScalarProduct(a.r2c2, b.r2c3), getScalarProduct(a.r2c3, b.r3c3)));
    combinedMatrix.r3c0 = getScalarSum(getScalarSum(getScalarProduct(a.r3c0, b.r0c0), getScalarProduct(a.r3c1, b.r1c0)), getScalarSum(getScalarProduct(a.r3c2, b.r2c0), getScalarProduct(a.r3c3, b.r3c0)));
    combinedMatrix.r3c1 = getScalarSum(getScalarSum(getScalarProduct(a.r3c0, b.r0c1), getScalarProduct(a.r3c1, b.r1c1)), getScalarSum(getScalarProduct(a.r3c2, b.r2c1), getScalarProduct(a.r3c3, b.r3c1)));
    combinedMatrix.r3c2 = getScalarSum(getScalarSum(getScalarProduct(a.r3c0, b.r0c2), getScalarProduct(a.r3c1, b.r1c2)), getScalarSum(getScalarProduct(a.r3c2, b.r2c2), getScalarProduct(a.r3c3, b.r3c2)));
    combinedMatrix.r3c3 = getScalarSum(getScalarSum(getScalarProduct(a.r3c0, b.r0c3), getScalarProduct(a.r3c1, b.r1c3)), getScalarSum(getScalarProduct(a.r3c2, b.r2c3), getScalarProduct(a.r3c3, b.r3c3)));
    return combinedMatrix;
}

void printBigMatrix(ComplexMatrix4 a) {
    printf("{\n{%.3lf + i(%.3lf), \t", a.r0c0.real, a.r0c0.imag);
    printf("%.3lf + i(%.3lf), \t", a.r0c1.real, a.r0c1.imag);
    printf("%.3lf + i(%.3lf), \t", a.r0c2.real, a.r0c2.imag);
    printf("%.3lf + i(%.3lf)}, \n", a.r0c3.real, a.r0c3.imag);
    printf("{%.3lf + i(%.3lf), \t", a.r1c0.real, a.r1c0.imag);
    printf("%.3lf + i(%.3lf), \t", a.r1c1.real, a.r1c1.imag);
    printf("%.3lf + i(%.3lf), \t", a.r1c2.real, a.r1c2.imag);
    printf("%.3lf + i(%.3lf)},\n", a.r1c3.real, a.r1c3.imag);
    printf("{%.3lf + i(%.3lf), \t", a.r2c0.real, a.r2c0.imag);
    printf("%.3lf + i(%.3lf), \t", a.r2c1.real, a.r2c1.imag);
    printf("%.3lf + i(%.3lf), \t", a.r2c2.real, a.r2c2.imag);
    printf("%.3lf + i(%.3lf)},\n", a.r2c3.real, a.r2c3.imag);
    printf("{%.3lf + i(%.3lf), \t", a.r3c0.real, a.r3c0.imag);
    printf("%.3lf + i(%.3lf), \t", a.r3c1.real, a.r3c1.imag);
    printf("%.3lf + i(%.3lf), \t", a.r3c2.real, a.r3c2.imag);
    printf("%.3lf + i(%.3lf)}\n}\n", a.r3c3.real, a.r3c3.imag);
}

ComplexMatrix4 recombine(
    ComplexMatrix2 l0, ComplexMatrix2 l1,
    ComplexMatrix2 d0,  ComplexMatrix2 d1,
    ComplexMatrix2 r0, ComplexMatrix2 r1
) {
    ComplexMatrix4 leftM = {0};
    
    leftM.r0c0 = l0.r0c0; leftM.r0c1 = l0.r0c1;
    leftM.r1c0 = l0.r1c0; leftM.r1c1 = l0.r1c1;
    
    leftM.r2c2 = l1.r0c0; leftM.r2c3 = l1.r0c1;
    leftM.r3c2 = l1.r1c0; leftM.r2c3 = l1.r1c1;
    
    ComplexMatrix4 rightM = {0};
    
    rightM.r0c0 = d0.r0c0; rightM.r0c1 = d0.r0c1;
    rightM.r1c0 = d0.r1c0; rightM.r1c1 = d0.r1c1;
    
    rightM.r2c2 = d1.r0c0; rightM.r2c3 = d1.r0c1;
    rightM.r3c2 = d1.r1c0; rightM.r2c3 = d1.r1c1;
    
    ComplexMatrix4 middleM = {0};
    
    middleM.r0c0 = d0.r0c0; middleM.r0c1 = d0.r0c1;
    middleM.r1c0 = d0.r1c0; middleM.r1c1 = d0.r1c1;
    
    middleM.r2c2 = d0.r0c0; middleM.r2c3 = d0.r0c1;
    middleM.r3c2 = d0.r1c0; middleM.r3c3 = d0.r1c1;
    
    middleM.r0c2 = d1.r0c0; middleM.r0c3 = d1.r0c1;
    middleM.r1c2 = d1.r1c0; middleM.r1c3 = d1.r1c1;
    
    middleM.r2c0 = getScalarFactor(-1,d1.r0c0); middleM.r2c1 = getScalarFactor(-1,d1.r0c1);
    middleM.r3c0 = getScalarFactor(-1,d1.r1c0); middleM.r3c1 = getScalarFactor(-1,d1.r1c1);
    
    printf("left decomp:\n");
    printBigMatrix(leftM);
    printf("middle decomp:\n");
    printBigMatrix(middleM);
    printf("right decomp:\n");
    printBigMatrix(rightM);
    
    ComplexMatrix4 combinedMatrix = get4x4MatrixProduct(leftM, get4x4MatrixProduct(middleM, rightM));
    return combinedMatrix;
}

void mytest() {
    
    
    printf("TESTING SVD\n\n");
    
    // check SVD  
    /*
    ComplexMatrix2 m = {
        .r0c0 = {.real=1,.imag=2}, .r0c1 = {.real=2,.imag=2},
        .r1c0 = {.real=3,.imag=3}, .r1c1 = {.real=4,.imag=4}
    };
    */
    ComplexMatrix2 m = {
        .r0c0 = {.real=1,.imag=1}, .r0c1 = {.real=2,.imag=2},
        .r1c0 = {.real=5,.imag=5}, .r1c1 = {.real=6,.imag=6}
    };
    
    ComplexMatrix2 l, d, r;
    singularValueDecompose(m, &l, &d, &r);
    printf("m:\n");
    printMatrix(m);
    printf("l:\n");
    printMatrix(l);
    printf("\nd:\n");
    printMatrix(d);
    printf("\nr:\n");
    printMatrix(r);
    printf("recombined:\n");
    printMatrix(
        getMatrixProduct(l,
            getMatrixProduct(d,r))
    );
    
    
    
    printf("\n\nTESTING COSINE-SINE \n\n");
    
    ComplexMatrix4 u;
    u.r0c0 = (Complex) {.real=1,.imag=1};
    u.r0c1 = (Complex) {.real=2,.imag=2};
    u.r0c2 = (Complex) {.real=3,.imag=3};
    u.r0c3 = (Complex) {.real=4,.imag=4};
    u.r1c0 = (Complex) {.real=5,.imag=5};
    u.r1c1 = (Complex) {.real=6,.imag=6};
    u.r1c2 = (Complex) {.real=7,.imag=7};
    u.r1c3 = (Complex) {.real=8,.imag=8};
    u.r2c0 = (Complex) {.real=9,.imag=9};
    u.r2c1 = (Complex) {.real=10,.imag=10};
    u.r2c2 = (Complex) {.real=11,.imag=11};
    u.r2c3 = (Complex) {.real=12,.imag=12};
    u.r3c0 = (Complex) {.real=13,.imag=13};
    u.r3c1 = (Complex) {.real=14,.imag=14};
    u.r3c2 = (Complex) {.real=15,.imag=15};
    u.r3c3 = (Complex) {.real=16,.imag=16};
    
    ComplexMatrix2 l0, l1, d0, d1, r0, r1;
    cosineSineDecompose(u, &l0, &l1, &d0, &d1, &r0, &r1);
    
    printf("input:\n");
    printBigMatrix(u);
    printf("\n");
    
    // recombine the above
    ComplexMatrix4 v = recombine(l0, l1, d0, d1, r0, r1);
    printf("recombined:\n");
    printBigMatrix(v);

}


#ifdef __cplusplus
}
#endif
