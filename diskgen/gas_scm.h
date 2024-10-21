#pragma once
#include "coord.h"
#include "galaxymodel_base.h"
#include "math_base.h"
#include "math_linalg.h"
#include "potential_base.h"
#include "potential_cylspline.h"
#include "particles_base.h"
#include "math_base.h"
#include <cstdlib>
#include <string>


using potential::PtrDensity;
using potential::PtrPotential;


/** Construct a self-consistent isothermal exponential gas disk with Wang (2010)'s method */
class GasDisk {
		public:
		GasDisk(double S0, double R0, double temp, double mu=1.23, double kB_over_mp=8.2543997567e-03);

		// Compute the velocity of gas particles
		void computeVelocity(coord::PosVelCyl& point, const PtrPotential& totalPot) const;		
		void computeVelocity(coord::PosVelCar& point, const PtrPotential& totalPot) const;
		void computeVelocity(coord::PosVelSph& point, const PtrPotential& totalPot) const;

		// Return denisty pointer of the gas disk
		PtrDensity createDensity(const PtrPotential& totalPot);


private:
		const double S0;         /// Surface density
		const double R0;         /// Scale radius
		const double temp;       /// temperature in K
														 ///
		const double gamma = 1;  /// adiabatic index (fixed to 1 for isothermal gas)
		const double mu; 	       /// mean atomic weight
		const double kB_over_mp; /// Boltzmann constant over proton mass (in intUnits.velocity^2/K)
		const double c2;         /// sound speed squared
		double int_energy;       /// specific internal energy


		/** Phi_z(R, z) := Phi(R, z) - Phi(R,0) */
		static double Phi_z(const coord::PosCyl &pos, const PtrPotential& totalPot);

		/** calcuate density */
		double density(const coord::PosCyl &pos, const PtrPotential& totalPot) const;

		/** Helper class for the integral in the density function */
		class ExpPhic2 : public math::IFunctionNoDeriv {
			public:
				ExpPhic2(const coord::PosCyl &pos, const PtrPotential& totalPot, double c2);

			private:
				const coord::PosCyl pos;
				const PtrPotential& totalPot;
				const double c2;

				virtual double value(const double z) const;
		};


		/** Density class for the gas disk */
		class IsothermalGasDisk: public potential::BaseDensity {
			public:
				IsothermalGasDisk(GasDisk* gasDisk, const PtrPotential& totalPot);
				virtual coord::SymmetryType symmetry() const { return sym_type; }
				virtual std::string name() const { return myName(); }
				static std::string myName() { return "IsothermalGasDisk"; }

			
			private:
				GasDisk* gasDisk;
				const PtrPotential& totalPot;
				const coord::SymmetryType sym_type = coord::ST_AXISYMMETRIC;

				virtual double densityCyl(const coord::PosCyl &pos, double time) const; 
				virtual double densityCar(const coord::PosCar &pos, double time) const;
				virtual double densitySph(const coord::PosSph &pos, double time) const;
		};

};

/// Helper class for computing the density at a given point in Cartesian coordinates
// Used for the sampling of the gas particles
class DensityIntegrandCar : public math:: IFunctionNdim {
	public :
		DensityIntegrandCar(const potential::BaseDensity& _dens);

		virtual void eval(const double vars[], double values[]) const;
    virtual unsigned int numVars() const { return  3; }
    virtual unsigned int numValues() const { return 1; }

	private:
		 const potential::BaseDensity& dens;
};


/// Helper class for sampling position and velocity from a DF (not scaled)
class DFIntegrandCar : public math::IFunctionNdim {
	public:
		DFIntegrandCar(const galaxymodel::GalaxyModel& _model);
		virtual void eval(const double vars[], double values[]) const;

    virtual unsigned int numVars() const { return  6; }
    virtual unsigned int numValues() const { return 1; }

	private:
		const galaxymodel::GalaxyModel& model;
};


/// Selection function for the local box area
/// Not used now because samplign in scaled coordinates does not work
class SelectionFunctionLocalBox: public galaxymodel::BaseSelectionFunction {
	public:
		SelectionFunctionLocalBox(const std::vector<double>& lower_pos, const std::vector<double>& upper_pos);

		virtual double value(const coord::PosVelCar& point) const;

	private:
		const std::vector<double> lower_pos, upper_pos;;

		virtual bool contains(const coord::PosVelCar &point) const;
};


/// compute the mass of a density component within a given box
double boxMass(const PtrDensity ptrDens, const double lower[], const double upper[]);


/// sample position from a density distribution
math::Matrix<double>  sampleDensityCar(const PtrDensity ptrDens, const size_t numPoints, const double lower[], const double upper[], double* totalMass, double* errorMass);


/// sample position and velocity from a DF
math::Matrix<double>  samplePosVelCar(const galaxymodel::GalaxyModel model, const size_t numPoints, const double lower_pos[], const double upper_pos[], double* totalMass, double* errorMass);


/// sample particles from self-consistent model 
particles::ParticleArrayCar sampleParticles(
		const galaxymodel::GalaxyModel model, const double partMass,
		const double lower_pos[], const double upper_pos[]);


/// sample particles from the gas disk density component and assign velocities
particles::ParticleArrayCar sampleParticles(
		const PtrDensity& ptrDens, const PtrPotential& totalPot, const GasDisk& gasDisk,
		const double partMass,
		const double lower_pos[], const double upper_pos[]);
