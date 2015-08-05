/** \file    potential_sphharm.h
    \brief   potential approximations based on spherical-harmonic expansion
    \author  Eugene Vasiliev
    \date    2010-2015
**/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_spline.h"
#include <vector>

namespace potential {

/** parent class for all potential expansions based on spherical harmonics for angular variables **/
class BasePotentialSphericalHarmonic: public BasePotentialSph {
public:
    BasePotentialSphericalHarmonic(unsigned int _Ncoefs_angular) :
        BasePotentialSph(), Ncoefs_angular(_Ncoefs_angular) {}
    virtual SymmetryType symmetry() const { return mysymmetry; };
    unsigned int getNumCoefsAngular() const { return Ncoefs_angular; }

protected:
    unsigned int Ncoefs_angular;         ///< l_max, the order of angular expansion (0 means spherically symmetric model)
    int lmax, lstep, mmin, mmax, mstep;  ///< range of angular coefficients used for given symmetry

    /// assigns the range for angular coefficients based on mysymmetry
    void setSymmetry(SymmetryType sym);

    /** The function that computes spherical-harmonic coefficients for potential 
        and its radial (first/second) derivative at given radius.
        Must be implemented in derived classes, and is used in evaluation of potential and forces; 
        unnecessary coefficients are indicated by passing NULL for coefs** and should not be computed.
    */
    virtual void computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const = 0;

private:
    SymmetryType mysymmetry;    ///< may have different type of symmetry

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;
};

/** basis-set expansion on the Zhao(1996) basis set (alpha models) **/
class BasisSetExp: public BasePotentialSphericalHarmonic {
public:
    /// init coefficients from a discrete point mass set
    BasisSetExp(double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
        const particles::PointMassArray<coord::PosSph> &points, SymmetryType _sym=ST_TRIAXIAL);

    /// load coefficients from stored values
    BasisSetExp(double _Alpha, const std::vector< std::vector<double> > &coefs);

    /// init potential coefficients from an analytic mass model
    BasisSetExp(double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
        const BaseDensity& density);
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "BSE"; };

    //  get functions:
    /// return BSE coefficients array
    void getCoefs(std::vector< std::vector<double>  > *coefsArray) const
    { if(coefsArray!=0) *coefsArray=SHcoefs; };

    /// return the shape parameter of basis set
    double getAlpha() const { return Alpha; };

    /// return the number of radial basis functions
    unsigned int getNumCoefsRadial() const { return Ncoefs_radial; }

    /// a faster estimate of M(r) from the l=0 harmonic only
    double enclosedMass(const double radius, const double /*rel_toler*/) const;

private:
    unsigned int Ncoefs_radial;                 ///< number of radial basis functions [ =SHcoefs.size() ]
    std::vector<std::vector<double> > SHcoefs;  ///< array of coefficients A_nlm of potential expansion
    double Alpha;                               ///< shape parameter controlling inner and outer slopes of basis functions

    /// compute angular expansion coefs at the given radius
    virtual void computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const;

    /// compute coefficients from a discrete point mass set; 
    /// if Alpha=0 then it is computed automatically from the data
    void prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph>& points);

    /// compute coefficients from a smooth mass profile; 
    /// if Alpha=0 then it is chosen automatically from density->getGamma()
    void prepareCoefsAnalytic(const BaseDensity& density);

    /// assigns symmetry class if some coefficients are (near-)zero
    void checkSymmetry();
};

/** spherical-harmonic expansion of potential with coefficients being spline functions of radius **/
class SplineExp: public BasePotentialSphericalHarmonic
{
public:
    /// init potential from N point masses with given assumed symmetry type, 
    /// may also provide desired grid radii (otherwise assigned automatically)
    SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
        const particles::PointMassArray<coord::PosSph> &points, 
        SymmetryType _sym=ST_TRIAXIAL, double smoothfactor=0, 
        const std::vector<double>  *_gridradii=0);

    /// init potential from stored SHE coefficients at given radii
    SplineExp(const std::vector<double>  &_gridradii, const std::vector< std::vector<double>  > &_coefs);

    /// init potential from analytic mass model, 
    /// may also provide desired grid radii (if not given, assign automatically)
    SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
        const BaseDensity& density, const std::vector<double>  *_gridradii=0);

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "Spline"; };

    // get functions
    unsigned int getNumCoefsRadial() const { return Ncoefs_radial; }   ///< return the number of radial points in the spline (excluding r=0)

    /** compute SHE coefficients at given radii.
        \param[in] useNodes tells whether the radii in question are 
                   the original spline nodes (true) or user-specified radii (true);
        \param[in,out] radii: if useNodes=true, this array is filled with spline nodes [output parameter, but must point to an existing array], 
                   otherwise the coefficients are computed at the supplied radii [input parameter];
        \param[out] coefsArray is filled by the computed coefficients (must point to an existing array)
    */
    void getCoefs(std::vector<double> *radii, std::vector< std::vector<double> > *coefsArray, bool useNodes=true) const;

    /// a faster estimate of M(r) from the l=0 harmonic only
    double enclosedMass(const double radius, const double /*rel_toler*/) const;

private:
    unsigned int Ncoefs_radial;              ///< number of radial coefficients (excluding the one at r=0)
    std::vector<double> gridradii;           ///< defines nodes of radial grid in splines
    double minr, maxr;                       ///< definition range of splines; extrapolation beyond this radius 
    double ascale;                           ///< value of scaling radius for non-spherical expansion coefficients which are tabulated as functions of log(r+ascale)
    double gammain,  coefin;                 ///< slope and coef. for extrapolating potential inside minr (spherically-symmetric part, l=0)
    double gammaout, coefout, der2out;       ///< slope and coef. for extrapolating potential outside maxr (spherically-symmetric part, l=0)
    double potcenter, potmax, potminr;       ///< (abs.value) potential in the center (for transformation of l=0 spline), at the outermost spline node, and at 1st spline node
    std::vector<math::CubicSpline> splines;  ///< spline coefficients at each harmonic
    std::vector<double> slopein, slopeout;   ///< slope of coefs for l>0 for extrapolating inside rmin/outside rmax

    /// reimplemented function to compute the coefficients of angular expansion of potential at the given radius
    virtual void computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const;

    /// assigns symmetry class if some coefficients are (near-)zero.
    /// called on an intermediate step after computing coefs but before initializing splines
    void checkSymmetry(const std::vector< std::vector<double> > &coefsArray);

    /// create spline objects for all non-zero spherical harmonics from the supplied radii and coefficients.
    /// radii should have "Ncoefs_radial" elements and coefsArray - "Ncoefs_radial * (Ncoefs_angular+1)^2" elements
    void initSpline(const std::vector<double>& radii, const std::vector<std::vector<double> >& coefsArray);

    /** calculate (non-smoothed) spherical harmonic coefs for a discrete point mass set.
        \param[in]  points contains the positions and masses of particles that serve as the input.
        \param[out] outradii will contain radii of particles sorted in ascending order,
        \param[out] outcoefs will contain coefficients of SH expansion for all particles
        in a 'swapped' order, namely: the size of this array is numCoefs^2,
        and elements of this array are `outcoefs[coefIndex][particleIndex]`
        this is done to save memory by not allocating arrays for unused coefficients.
    */
    void computeCoefsFromPoints(const particles::PointMassArray<coord::PosSph>& points, 
        std::vector<double>& outradii, std::vector<std::vector<double> >& outcoefs);

    /** create smoothing splines from the coefficients computed at each particle's radius.
        \param[in] points is the array of particle coordinates and masses
        \param[in] userradii may contain the radial grid to construct the splines; 
                   if it is NULL then the radial grid is constructed automatically 
                   taking into account the range of radii covered by source points.
        \param[in] smoothfactor determines how much smoothing is applied to the spline 
                   (good results are obtained for smoothfactor=1-2). 
    */
    void prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph>& points, 
        double smoothfactor, const std::vector<double> * userradii);

    /** compute expansion coefficients from an analytical mass profile.
        \param[in] density is the density profile to be approximated
        \param[in] srcradii may contain the radial grid; if NULL then it is assigned automatically.
    */
    void prepareCoefsAnalytic(const BaseDensity& density, const std::vector<double>  *srcradii);

    /// evaluate value and optionally up to two derivatives of l=0 coefficient, 
    /// taking into account extrapolation beyond the grid definition range and log-scaling of splines
    void coef0(double r, double *val, double *der, double *der2) const;

    /// evaluate value, and optionally first and second derivative of l>0 coefficients 
    /// (lm is the combined index of angular harmonic >0); corresponding values for 0th coef must be known
    void coeflm(unsigned int lm, double r, double xi, double *val, double *der, double *der2, 
        double c0val, double c0der=0, double c0der2=0) const;
};

}  // namespace