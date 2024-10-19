/** \file    example_self_consistent_model.cpp
    \author  Eugene Vasiliev
    \date    2015-2017

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
    We create a four-component galaxy with disk, bulge and halo components defined by their DFs,
    and a static density profile of gas disk.
    Then we perform several iterations of recomputing the density profiles of components from their DFs
    and recomputing the total potential.
    Finally, we create N-body representations of all mass components: dark matter halo,
    stars (bulge, thin and thick disks and stellar halo combined), and gas disk.

    An equivalent Python example is given in pytests/example_self_consistent_model.py
*/
#include "actions_base.h"
#include "actions_staeckel.h"
#include "coord.h"
#include "galaxymodel_base.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel_velocitysampler.h"
#include "df_factory.h"
#include "math_base.h"
#include "math_linalg.h"
#include "math_sample.h"
#include "particles_base.h"
#include "potential_base.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "potential_multipole.h"
#include "potential_cylspline.h"
#include "potential_utils.h"
#include "particles_io.h"
#include "math_core.h"
#include "math_base.h"
#include "math_spline.h"
#include "units.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <string>

using potential::PtrDensity;
using potential::PtrPotential;

/** Construct a self-consistent isothermal exponential gas disk with Wang (2010)'s method
 */
class GasDisk {
		public:
		GasDisk(double S0, double R0, double temp, double mu=1.23, double kB_over_mp=8.2543997567e-03) :
			S0(S0), R0(R0), temp(temp), mu(mu), kB_over_mp(kB_over_mp), c2(kB_over_mp*temp/mu) {
				int_energy = c2;
				if (gamma != 1.0) {
					int_energy /= (gamma - 1.0);
				}
			}

		void computeVelocity(coord::PosVelCyl& point, const PtrPotential& totalPot) const {
			const double dR = 0.01;
			double rho_z0 = density(point, totalPot);
			double drho = density(coord::PosCyl(point.R + 0.5*dR, 0., 0.), totalPot);;
			drho -= density(coord::PosCyl(point.R - 0.5*dR, 0., 0.), totalPot);;
			double velCirc2 = pow_2(potential::v_circ(*totalPot, point.R));
			double vphi2 = velCirc2 + c2*point.R/rho_z0*drho/dR;

			if (vphi2 > 0) {point.vphi = sqrt(vphi2);} else {point.vphi = 0;}
			point.vR = 0.;
			point.vz = 0.;
		}
		void computeVelocity(coord::PosVelCar& point, const PtrPotential& totalPot) const {
			coord::PosVelCyl cyl = coord::toPosVelCyl(point);
			computeVelocity(cyl, totalPot);
			point = coord::toPosVelCar(cyl);
		}
		void computeVelocity(coord::PosVelSph& point, const PtrPotential& totalPot) const {
			coord::PosVelCyl cyl = coord::toPosVelCyl(point);
			computeVelocity(cyl, totalPot);
			point = coord::toPosVelSph(cyl);
		}


		PtrDensity createDensity(const PtrPotential& totalPot) {
			return PtrDensity(new IsothermalGasDisk(this, totalPot));
		}


private:
		const double S0;         /// Surface density
		const double R0;         /// Scale radius
		const double temp;       /// temperature in K
														 ///
		const double gamma = 1;  /// adiabatic index (fixed to 1 for isothermal gas)
		const double mu; 	       /// mean atomic weight
		const double kB_over_mp; /// Boltzmann constant over proton mass (in intUnits.velocity^2/K)
		const double c2;               /// sound speed squared
		double int_energy;       /// specific internal energy


		/** Phi_z(R, z) := Phi(R, z) - Phi(R,0) */
		static double Phi_z(const coord::PosCyl &pos, const PtrPotential& totalPot) {
			double pot_value = totalPot->value(pos);
			pot_value -= totalPot->value(coord::PosCyl(pos.R, 0, 0));
			return pot_value;
		}

		/** calcuate density */
		double density(const coord::PosCyl &pos, const PtrPotential& totalPot) const {
			// TODO : automatically set the upper limit of the integral
			double I = math::integrate(ExpPhic2(pos, totalPot, c2), 0, 10, 1e-3);
			double rho0 = 0.5 * S0 * exp(-pos.R / R0) / I;
			return rho0 * exp(- Phi_z(pos, totalPot)/ c2);
		}

		/** Helper class for the integral in the density function */
		class ExpPhic2 : public math::IFunctionNoDeriv {
			public:
				ExpPhic2(const coord::PosCyl &pos, const PtrPotential& totalPot, double c2): pos(pos), totalPot(totalPot), c2(c2){};

			private:
				const coord::PosCyl pos;
				const PtrPotential& totalPot;
				const double c2;

				virtual double value(const double z) const {
					return exp(- Phi_z(coord::PosCyl(pos.R, z, 0), totalPot) / c2);
				}
		};

		class IsothermalGasDisk: public potential::BaseDensity {
			public:
				IsothermalGasDisk(GasDisk* gasDisk,
						const PtrPotential& totalPot):
				 	gasDisk(gasDisk), totalPot(totalPot){};
				virtual coord::SymmetryType symmetry() const { return sym_type; }
				virtual std::string name() const { return myName(); }
				static std::string myName() { return "IsothermalGasDisk"; }

			
			private:
				GasDisk* gasDisk;
				const PtrPotential& totalPot;
				const coord::SymmetryType sym_type = coord::ST_AXISYMMETRIC;

				virtual double densityCyl(const coord::PosCyl &pos, double time) const { 
					return gasDisk->density(pos, totalPot);
				}
				virtual double densityCar(const coord::PosCar &pos, double time) const { 
					return densityCyl(toPosCyl(pos), time);
				}
				virtual double densitySph(const coord::PosSph &pos, double time) const {
					return densityCyl(toPosCyl(pos), time);
				}
		};

};

/// Helper class for computing the density at a given point in Cartesian coordinates
class DensityIntegrandCar : public math:: IFunctionNdim {
	public :
		DensityIntegrandCar(const potential::BaseDensity& _dens) : dens(_dens) {}

		virtual void eval(const double vars[], double values[]) const {
			coord::PosCar pos(vars[0], vars[1], vars[2]);
			values[0] = dens.density(pos, /*time*/0);
			if (!isFinite(values[0]) || values[0] < 0) values[0] = 0;
		}

    virtual unsigned int numVars() const { return  3; }
    virtual unsigned int numValues() const { return 1; }

	private:
		 const potential::BaseDensity& dens;
};

/// compute the mass of a density component within a given box
double boxMass(const PtrDensity ptrDens, const double lower[], const double upper[]) {
	double mass = 0;
	math::integrateNdim(DensityIntegrandCar(*ptrDens),
			lower, upper, 1e-4, 1e6, &mass);
	return mass;
}

/// sample particles from a density component within a given box
particles::ParticleArrayCar sampleDensityCar(const PtrDensity ptrDens, const size_t numPoints,
  const double lower[], const double upper[]) {
	 math::Matrix<double> result;
	 double totalMass, errorMass;
	 
	 math::sampleNdim(DensityIntegrandCar(*ptrDens), lower, upper, numPoints, result, NULL, &totalMass, &errorMass);

	 const double pointMass = totalMass / result.rows();
	 particles::ParticleArray<coord::PosVelCar> points;
	 for(size_t i=0; i<result.rows(); i++) {
		 points.add(coord::PosVelCar(result(i,0), result(i,1), result(i,2), 0, 0, 0), pointMass);
	 }

	 return points;
};


/// Helper class for sampling position and velocity from a DF (not scaled)
class DFIntegrandCar : public math::IFunctionNdim {
	public:
		DFIntegrandCar(const galaxymodel::GalaxyModel& _model): model(_model){}

		virtual void eval(const double vars[], double values[]) const {
			actions::Actions act = model.actFinder.actions(
					coord::toPosVelCyl(coord::PosVelCar(vars[0], vars[1], vars[2], vars[3], vars[4], vars[5]))
					);
			values[0] = model.distrFunc.value(act);
			if (!isFinite(values[0]) || values[0] < 0) values[0] = 0;
		}

    virtual unsigned int numVars() const { return  6; }
    virtual unsigned int numValues() const { return 1; }

	private:
		const galaxymodel::GalaxyModel& model;
};

/// sample particles from a DF within a given box
particles::ParticleArrayCar samplePosVelCar(const galaxymodel::GalaxyModel model, const double partMass,
  const double lower_pos[], const double upper_pos[]) {
	double Phi;
	model.potential.eval(coord::PosCar(0.5*(lower_pos[0]+upper_pos[0]),
																		 0.5*(lower_pos[1]+upper_pos[1]),
																		 0.5*(lower_pos[2]+upper_pos[2])),
			&Phi);
	double vesc = sqrt(-2.*Phi);
	double lower[6] = {lower_pos[0], lower_pos[1], lower_pos[2], -vesc*0.57, -vesc*0.57, -vesc*0.57};
	double upper[6] = {upper_pos[0], upper_pos[1], upper_pos[2],  vesc*0.57,  vesc*0.57,  vesc*0.57};
	math::Matrix<double> result;
	double totalMass, errorMass;

	math::sampleNdim(DFIntegrandCar(model), lower, upper, 1e6, result, NULL, &totalMass, &errorMass);
	if (totalMass < partMass) { return particles::ParticleArrayCar(); }

	size_t numPoints = (size_t)(totalMass / partMass);
	if (numPoints > result.rows()) {
		math::sampleNdim(DFIntegrandCar(model), lower, upper, numPoints, result, NULL, &totalMass, &errorMass);
	}
	particles::ParticleArray<coord::PosVelCar> points;
	for(size_t i=0; i<numPoints; i++) {
		points.add(coord::PosVelCar(result(i,0), result(i,1), result(i,2),
																result(i,3), result(i,4), result(i,5)),
				partMass);
	}

	 return points;
};

// define internal unit system - arbitrary numbers here! the result should not depend on their choice
const units::InternalUnits intUnits(1*units::Kpc, 977.7922216807891*units::Myr);

// define external unit system describing the data (including the parameters in INI file)
const units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 1.*units::kms, 1.*units::Msun);

// used for outputting the velocity distribution (the value is read from the ini file)
double solarRadius = NAN;

// various auxiliary functions for printing out information are non-essential
// for the modelling itself; the essential workflow is contained in main()

/// print the rotation curve for a collection of potential components into a text file
void writeRotationCurve(const std::string& filename, const std::vector<PtrPotential>& potentials)
{
	std::ofstream strm(filename.c_str());
	strm << "# radius[Kpc]\tv_circ,total[km/s]\tdisk\tbulge\thalo\n";
	// print values at certain radii, expressed in units of Kpc
	std::vector<double> radii = math::createExpGrid(81, 0.01, 100);
	for(unsigned int i=0; i<radii.size(); i++) {
		strm << radii[i];  // output radius in kpc
		double v2sum = 0;  // accumulate squared velocity in internal units
		double r_int = radii[i] * intUnits.from_Kpc;  // radius in internal units
		std::string str;
		for(unsigned int i=0; i<potentials.size(); i++) {
			double vc = v_circ(*potentials[i], r_int);
			v2sum += pow_2(vc);
			str += "\t" + utils::toString(vc * intUnits.to_kms);  // output in km/s
		}
		strm << '\t' << (sqrt(v2sum) * intUnits.to_kms) << str << '\n';
	}
}

/// print surface density profiles to a file
void writeSurfaceDensityProfile(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
	std::cout << "Writing surface density profile\n";
	std::vector<double> radii;
	// convert radii to internal units
	for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
		radii.push_back(r * intUnits.from_Kpc);
	int nr = radii.size();
	int nc = model.distrFunc.numValues();  // number of DF components
	std::vector<double> surfDens(nr*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for(int ir=0; ir<nr; ir++) {
		computeMoments(model, coord::PosProj(radii[ir],0),
				&surfDens[ir*nc], NULL, NULL, /*separate*/ true);
	}

	std::ofstream strm(filename.c_str());
	strm << "# Radius[Kpc]\tThinDisk\tThickDisk\tStellarHalo:SurfaceDensity[Msun/pc^2]\n";
	for(int ir=0; ir<nr; ir++) {
		strm << radii[ir] * intUnits.to_Kpc;
		for(int ic=0; ic<nc; ic++)
			strm << '\t' << surfDens[ir*nc+ic] * intUnits.to_Msun_per_pc2;
		strm << '\n';
	}
}

/// print vertical density profile for several sub-components of the stellar DF
void writeVerticalDensityProfile(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
	std::cout << "Writing vertical density profile\n";
	std::vector<double> heights;
	// convert height to internal units
	for(double h=0; h<=8; h<1.5 ? h+=0.125 : h+=0.5)
		heights.push_back(h * intUnits.from_Kpc);
	double R = solarRadius * intUnits.from_Kpc;
	int nh = heights.size();
	int nc = model.distrFunc.numValues();
	std::vector<double> dens(nh*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for(int ih=0; ih<nh; ih++) {
		computeMoments(model, coord::PosCar(R,0,heights[ih]),
				&dens[ih*nc], NULL, NULL, /*separate*/ true);
	}

	std::ofstream strm(filename.c_str());
	strm << "# z[Kpc]\tThinDisk\tThickDisk\tStellarHalo:Density[Msun/pc^3]\n";
	for(int ih=0; ih<nh; ih++) {
		strm << heights[ih] * intUnits.to_Kpc;
		for(int ic=0; ic<nc; ic++)
			strm << '\t' << dens[ih*nc+ic] * intUnits.to_Msun_per_pc3;
		strm << '\n';
	}
}

/// print velocity dispersion profiles in the equatorial plane as functions of radius to a file
void writeVelocityDispersionProfile(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
	std::cout << "Writing velocity dispersion profile\n";
	std::vector<double> radii;
	// convert radii to internal units
	for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
		radii.push_back(r * intUnits.from_Kpc);
	int nr = radii.size();
	int nc = model.distrFunc.numValues();  // number of DF components
	std::vector<coord::VelCar>  vel (nr*nc);
	std::vector<coord::Vel2Car> vel2(nr*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for(int ir=0; ir<nr; ir++) {
		computeMoments(model, coord::PosCar(radii[ir],0,0),
				NULL, &vel[ir*nc], &vel2[ir*nc], /*separate*/ true);
	}

	std::ofstream strm(filename.c_str());
	strm << "# Radius[Kpc]\tThinDisk:sigma_r\tsigma_phi\tsigma_z\tv_phi\t"
		"ThickDisk:sigma_r\tsigma_phi\tsigma_z\tv_phi\t"
		"StellarHalo:sigma_r\tsigma_phi\tsigma_z\tv_phi[km/s]\n";
	for(int ir=0; ir<nr; ir++) {
		strm << radii[ir] * intUnits.to_Kpc;
		for(int ic=0; ic<nc; ic++)
			strm << '\t' <<
				sqrt(vel2[ir*nc+ic].vx2) * intUnits.to_kms << '\t' <<
				sqrt(vel2[ir*nc+ic].vy2-pow_2(vel[ir*nc+ic].vy)) * intUnits.to_kms << '\t' <<
				sqrt(vel2[ir*nc+ic].vz2) * intUnits.to_kms << '\t' <<
				vel[ir*nc+ic].vy * intUnits.to_kms;
		strm << '\n';
	}
}
/// print velocity distributions at the given point to a file
void writeVelocityDistributions(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
	const coord::PosCar point(solarRadius * intUnits.from_Kpc, 0, 0.1 * intUnits.from_Kpc);
	std::cout << "Writing velocity distributions at "
		"(x=" << point.x * intUnits.to_Kpc << ", z=" << point.z * intUnits.to_Kpc << ")\n";
	// create grids in velocity space
	double v_max = 360 * intUnits.from_kms;
	// for simplicity use the same grid for all three dimensions
	std::vector<double> gridv = math::createUniformGrid(75, -v_max, v_max);
	std::vector<double> amplvx, amplvy, amplvz;
	double density;
	// compute the distributions
	const int ORDER = 3;
	math::BsplineInterpolator1d<ORDER> interp(gridv);
	galaxymodel::computeVelocityDistribution<ORDER>(model, point,
			gridv, gridv, gridv, /*output*/ &density, &amplvx, &amplvy, &amplvz);

	std::ofstream strm(filename.c_str());
	strm << "# V\tf(V_x)\tf(V_y)\tf(V_z) [1/(km/s)]\n";
	for(int i=-100; i<=100; i++) {
		double v = i*v_max/100;
		// unit conversion: the VDF has a dimension 1/V, so that \int f(V) dV = 1;
		// therefore we need to multiply it by 1/velocityUnit
		strm << utils::toString(v * intUnits.to_kms)+'\t'+
			utils::toString(interp.interpolate(v, amplvx) / intUnits.to_kms)+'\t'+
			utils::toString(interp.interpolate(v, amplvy) / intUnits.to_kms)+'\t'+
			utils::toString(interp.interpolate(v, amplvz) / intUnits.to_kms)+'\n';
	}
}

/// report progress after an iteration
void printoutInfo(const galaxymodel::SelfConsistentModel& model, const std::string& iteration)
{
	const potential::BaseDensity& compDisk = *model.components[0]->getDensity();
	const potential::BaseDensity& compBulge= *model.components[1]->getDensity();
	const potential::BaseDensity& compHalo = *model.components[2]->getDensity();
	const potential::BaseDensity& compGas  = *model.components[3]->getDensity();
	coord::PosCyl pt0(solarRadius * intUnits.from_Kpc, 0, 0);
	coord::PosCyl pt1(solarRadius * intUnits.from_Kpc, 1 * intUnits.from_Kpc, 0);
	std::cout <<
		"Disk total mass="      << (compDisk.totalMass()  * intUnits.to_Msun) << " Msun"
		", rho(Rsolar,z=0)="    << (compDisk.density(pt0) * intUnits.to_Msun_per_pc3) <<
		", rho(Rsolar,z=1kpc)=" << (compDisk.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
		"Bulge total mass="     << (compBulge.totalMass()  * intUnits.to_Msun) << " Msun\n"
		"Halo total mass="      << (compHalo.totalMass()  * intUnits.to_Msun) << " Msun"
		", rho(Rsolar,z=0)="    << (compHalo.density(pt0) * intUnits.to_Msun_per_pc3) <<
		", rho(Rsolar,z=1kpc)=" << (compHalo.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
		"Gas total mass="       << (compGas.totalMass()  * intUnits.to_Msun) << " Msun"
		", rho(Rsolar,z=0)="    << (compGas.density(pt0) * intUnits.to_Msun_per_pc3) <<
		", rho(Rsolar,z=1kpc)=" << (compGas.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
		"Potential at origin=-("<<
		(sqrt(-model.totalPotential->value(coord::PosCyl(0,0,0))) * intUnits.to_kms) << " km/s)^2"
		", total mass=" << (model.totalPotential->totalMass() * intUnits.to_Msun) << " Msun\n";

	// writeDensity("dens_disk_"+iteration, compDisk, extUnits);
	// writeDensity("dens_halo_"+iteration, compHalo, extUnits);
	// writePotential("potential_"+iteration, *model.totalPotential, extUnits);
	// std::vector<PtrPotential> potentials(3);
	// potentials[0] = dynamic_cast<const potential::Composite&>(*model.totalPotential).component(1);
	// potentials[1] = potential::Multipole::create(compBulge, /*lmax*/6, /*mmax*/0, /*gridsize*/25);
	// potentials[2] = potential::Multipole::create(compHalo,  /*lmax*/6, /*mmax*/0, /*gridsize*/25);
	// writeRotationCurve("rotcurve_"+iteration, potentials);
}

/// perform one iteration of the model
void doIteration(galaxymodel::SelfConsistentModel& model, GasDisk& gasDisk, int iterationIndex)
{
	std::cout << "\033[1;37mStarting iteration #" << iterationIndex << "\033[0m\n";
	bool error=false;
	try {
		/// replace the gas disk density function with a CylSpline approximation
		model.components[3] = galaxymodel::PtrComponent(new galaxymodel::ComponentStatic(
					potential::DensityAzimuthalHarmonic::create(
						*gasDisk.createDensity(model.totalPotential),
						/*mmax*/ 0,
						/*gridSizeR*/ 100, /*Rmin*/ 0.01, /*Rmax*/ 50,
						/*gridSizez*/ 100, /*zmin*/ 0.01, /*zmax*/ 20),
					true));

		/// update the N-body components
		doIteration(model);
	}
	catch(std::exception& ex) {
		error=true;  // report the error and allow to save the results of the last iteration
		std::cout << "\033[1;31m==== Exception occurred: \033[0m\n" << ex.what();
	}
	printoutInfo(model, "iter"+utils::toString(iterationIndex));
	if(error)
		exit(1);  // abort in case of problems
}


/// Selection function for the local box area
class SelectionFunctionLocalBox: public galaxymodel::BaseSelectionFunction {
	public:
		SelectionFunctionLocalBox(const std::vector<double>& lower_pos, const std::vector<double>& upper_pos):
			lower_pos(lower_pos), upper_pos(upper_pos) {};

		virtual double value(const coord::PosVelCar& point) const { 
			if (contains(point)) {return 1.;}
			else {return 0.;}
		}

	private:
		const std::vector<double> lower_pos, upper_pos;;

		virtual bool contains(const coord::PosVelCar &point) const {
			return point.x >= lower_pos[0] && point.x <= upper_pos[0] &&
				point.y >= lower_pos[1] && point.y <= upper_pos[1] &&
				point.z >= lower_pos[2] && point.z <= upper_pos[2];
		}

};



/// sample particles from self-consistent model 
particles::ParticleArrayCar sampleParticles(
		const galaxymodel::GalaxyModel model, const double partMass,
		const double lower_pos[], const double upper_pos[]) {
	particles::ParticleArrayCar	points;
	try {
		points = samplePosVelCar(model, partMass, lower_pos, upper_pos);

	} catch (const std::runtime_error& e) {
		 std::cerr <<  e.what() << std::endl;
		return particles::ParticleArrayCar();
		//if (std::string(e.what()) == "sampleNdim: function is identically zero inside the region") {
		//	std::cerr << "Caught specific runtime error: " << e.what() << std::endl;
		//	return points;
		//} else {
		//	std::cerr << "Unexpected runtime error: " << e.what() << std::endl;
		//	std::exit(EXIT_FAILURE); 
		//}
	} catch (const std::exception& e) {
		std::cerr << "Unexpected exception: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);  
	}

	return points;
}


/// sample particles from the gas disk density component and assign velocities
particles::ParticleArrayCar sampleParticles(
		const PtrDensity& ptrDens, const PtrPotential& totalPot, const GasDisk& gasDisk,
		const double partMass,
		const double lower_pos[], const double upper_pos[]) {

	double totalMass = boxMass(ptrDens, lower_pos, upper_pos);
	particles::ParticleArrayCar points;
	if (totalMass < partMass) {
		return particles::ParticleArrayCar();
	}

	try {
		size_t numPoints = (size_t)(totalMass / partMass);
	 points = sampleDensityCar(ptrDens, numPoints, lower_pos, upper_pos);
	} catch (const std::runtime_error& e) {
		return particles::ParticleArrayCar();
	} catch (const std::exception& e) {
		std::cerr << "Unexpected exception: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);  
	}


#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (size_t i = 0; i < points.size(); i++) {
		gasDisk.computeVelocity(points[i].first, totalPot);
		points[i].second = partMass;
	}

	return points;
}


int main()
{
	// read parameters from the INI file
	const std::string iniFileName = "../data/SCM.ini";
	utils::ConfigFile ini(iniFileName);
	utils::KeyValueMap
		iniPotenThinDisk = ini.findSection("Potential thin disk"),
		iniPotenThickDisk= ini.findSection("Potential thick disk"),
	  iniPotenGasDisk  = ini.findSection("Potential gas disk"),
		iniPotenBulge    = ini.findSection("Potential bulge"),
		iniPotenDarkHalo = ini.findSection("Potential dark halo"),
		iniDFThinDisk    = ini.findSection("DF thin disk"),
		iniDFThickDisk   = ini.findSection("DF thick disk"),
		iniDFStellarHalo = ini.findSection("DF stellar halo"),
		iniDFBulge       = ini.findSection("DF bulge"),
		iniDFDarkHalo    = ini.findSection("DF dark halo"),
		iniSCMDisk       = ini.findSection("SelfConsistentModel disk"),
		iniSCMBulge      = ini.findSection("SelfConsistentModel bulge"),
		iniSCMHalo       = ini.findSection("SelfConsistentModel halo"),
		iniSCM           = ini.findSection("SelfConsistentModel"),
	  iniGasParams     = ini.findSection("Gas parameters");
	if(!iniSCM.contains("rminSph")) {  // most likely file doesn't exist
		std::cout << "Invalid INI file " << iniFileName << "\n";
		return -1;
	}
	solarRadius = ini.findSection("Data").getDouble("SolarRadius", solarRadius);

	// set up parameters of the entire Self-Consistent Model
	galaxymodel::SelfConsistentModel model;
	model.rminSph         = iniSCM.getDouble("rminSph") * extUnits.lengthUnit;
	model.rmaxSph         = iniSCM.getDouble("rmaxSph") * extUnits.lengthUnit;
	model.sizeRadialSph   = iniSCM.getInt("sizeRadialSph");
	model.lmaxAngularSph  = iniSCM.getInt("lmaxAngularSph");
	model.RminCyl         = iniSCM.getDouble("RminCyl") * extUnits.lengthUnit;
	model.RmaxCyl         = iniSCM.getDouble("RmaxCyl") * extUnits.lengthUnit;
	model.zminCyl         = iniSCM.getDouble("zminCyl") * extUnits.lengthUnit;
	model.zmaxCyl         = iniSCM.getDouble("zmaxCyl") * extUnits.lengthUnit;
	model.sizeRadialCyl   = iniSCM.getInt("sizeRadialCyl");
	model.sizeVerticalCyl = iniSCM.getInt("sizeVerticalCyl");
	model.useActionInterpolation = iniSCM.getBool("useActionInterpolation");

	// initialize density profiles of various components
	std::vector<PtrDensity> densityStellarDisk(2);
	PtrDensity densityBulge    = potential::createDensity(iniPotenBulge,    extUnits);
	PtrDensity densityDarkHalo = potential::createDensity(iniPotenDarkHalo, extUnits);
	densityStellarDisk[0]      = potential::createDensity(iniPotenThinDisk, extUnits);
	densityStellarDisk[1]      = potential::createDensity(iniPotenThickDisk,extUnits);
	PtrDensity densityGasDisk  = potential::createDensity(iniPotenGasDisk,  extUnits);


	// add components to SCM - at first, all of them are static density profiles
	model.components.push_back(galaxymodel::PtrComponent(
				new galaxymodel::ComponentStatic(PtrDensity(
						new potential::CompositeDensity(densityStellarDisk)), true)));
	model.components.push_back(galaxymodel::PtrComponent(
				new galaxymodel::ComponentStatic(densityBulge, false)));
	model.components.push_back(galaxymodel::PtrComponent(
				new galaxymodel::ComponentStatic(densityDarkHalo, false)));
	model.components.push_back(galaxymodel::PtrComponent(
				new galaxymodel::ComponentStatic(densityGasDisk, true)));

	// initialize total potential of the model (first guess)
	updateTotalPotential(model);
	printoutInfo(model, "init");

	std::cout << "\033[1;33m**** STARTING MODELLING ****\033[0m\nInitial masses of density components: "
		"Mdisk="  << (model.components[0]->getDensity()->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mbulge=" << (densityBulge   ->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mhalo="  << (densityDarkHalo->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mgas="   << (densityGasDisk ->totalMass() * intUnits.to_Msun) << " Msun\n";

	// create the dark halo DF
	df::PtrDistributionFunction dfHalo = df::createDistributionFunction(
			iniDFDarkHalo, model.totalPotential.get(), /*density not needed*/NULL, extUnits);
	// same for the bulge
	df::PtrDistributionFunction dfBulge = df::createDistributionFunction(
			iniDFBulge, model.totalPotential.get(), NULL, extUnits);
	// same for the stellar components (thin/thick disks and stellar halo)
	df::PtrDistributionFunction dfThin =	df::createDistributionFunction(
				iniDFThinDisk, model.totalPotential.get(), NULL, extUnits);
	df::PtrDistributionFunction dfThick = df::createDistributionFunction(
			iniDFThickDisk, model.totalPotential.get(), NULL, extUnits);
	df::PtrDistributionFunction dfStellarHalo = df::createDistributionFunction(
			iniDFStellarHalo, model.totalPotential.get(), NULL, extUnits);
	std::vector<df::PtrDistributionFunction> dfStellarArray = {dfThin, dfThick, dfStellarHalo};
	df::PtrDistributionFunction dfStellar(new df::CompositeDF(dfStellarArray));

	// replace the static disk density component of SCM with a DF-based disk component
	model.components[0] = galaxymodel::PtrComponent(
			new galaxymodel::ComponentWithDisklikeDF(dfStellar, PtrDensity(),
				iniSCMDisk.getInt("mmaxAngularCyl"),
				iniSCMDisk.getInt("sizeRadialCyl"),
				iniSCMDisk.getDouble("RminCyl") * extUnits.lengthUnit,
				iniSCMDisk.getDouble("RmaxCyl") * extUnits.lengthUnit,
				iniSCMDisk.getInt("sizeVerticalCyl"),
				iniSCMDisk.getDouble("zminCyl") * extUnits.lengthUnit,
				iniSCMDisk.getDouble("zmaxCyl") * extUnits.lengthUnit));
	// same for the bulge
	model.components[1] = galaxymodel::PtrComponent(
			new galaxymodel::ComponentWithSpheroidalDF(dfBulge, potential::PtrDensity(),
				iniSCMBulge.getInt("lmaxAngularSph"),
				iniSCMBulge.getInt("mmaxAngularSph"),
				iniSCMBulge.getInt("sizeRadialSph"),
				iniSCMBulge.getDouble("rminSph") * extUnits.lengthUnit,
				iniSCMBulge.getDouble("rmaxSph") * extUnits.lengthUnit));
	// same for the halo
	model.components[2] = galaxymodel::PtrComponent(
			new galaxymodel::ComponentWithSpheroidalDF(dfHalo, potential::PtrDensity(),
				iniSCMHalo.getInt("lmaxAngularSph"),
				iniSCMHalo.getInt("mmaxAngularSph"),
				iniSCMHalo.getInt("sizeRadialSph"),
				iniSCMHalo.getDouble("rminSph") * extUnits.lengthUnit,
				iniSCMHalo.getDouble("rmaxSph") * extUnits.lengthUnit));

	// Initialize the gas disk
	GasDisk gasDisk(
			iniPotenGasDisk.getDouble("surfaceDensity") * intUnits.from_Msun_per_Kpc2,
			iniPotenGasDisk.getDouble("scaleRadius") * extUnits.lengthUnit,
			iniGasParams.getDouble("Temperature"));

	// do a few more iterations to obtain the self-consistent density profile for both disks
	for(int iteration=1; iteration<=5; iteration++){
		doIteration(model, gasDisk, iteration);
	}

	// final gas disk potential without CylSpline approximation
	model.components[3] = galaxymodel::PtrComponent(new galaxymodel::ComponentStatic(
				gasDisk.createDensity(model.totalPotential), true));
	printoutInfo(model, "Final");


	std::ifstream inputFile("../data/DomainCoord128.txt");
	std::vector<std::vector<double>> domain_data;
	std::string line;
	while (getline(inputFile, line)) {
		std::istringstream iss(line);
		std::vector<double> numbers;
		double num;

		for (int i = 0; i < 6; ++i) {
			if (iss >> num) {
				numbers.push_back(num);
			} else {
				std::cout << "Error in reading the input file" << std::endl;
				return 1;
			}
		}

		domain_data.push_back(numbers);
	}
	inputFile.close();

	//int idx = 0;
	//std::vector<double> lower_pos = {5 * extUnits.lengthUnit, 5 * extUnits.lengthUnit, 0 * extUnits.lengthUnit};
	//std::vector<double> upper_pos = {10 * extUnits.lengthUnit, 10 * extUnits.lengthUnit, 10 * extUnits.lengthUnit};

	//galaxymodel::GalaxyModel bulge(*model.totalPotential, *model.actionFinder, *dfBulge);
	//galaxymodel::GalaxyModel thinDisk(*model.totalPotential, *model.actionFinder, *dfThin);
	//galaxymodel::GalaxyModel thickDisk(*model.totalPotential, *model.actionFinder, *dfThick);
	//galaxymodel::GalaxyModel stellarHalo(*model.totalPotential, *model.actionFinder, *dfStellarHalo);
	//galaxymodel::GalaxyModel dmHalo(*model.totalPotential, *model.actionFinder, *dfHalo);
	//galaxymodel::GalaxyModel stellar(*model.totalPotential, *model.actionFinder, *dfStellar);
	//PtrDensity ptrDensGasDisk = gasDisk.createDensity(model.totalPotential);


	//particles::ParticleArrayCar par = sampleParticles(dmHalo, 1e6 * intUnits.from_Msun, lower_pos.data(), upper_pos.data());
	////particles::ParticleArrayCar par = sampleParticles(ptrDensGasDisk, model.totalPotential, gasDisk, 1e5 * intUnits.from_Msun, lower_pos.data(), upper_pos.data());
	//std::cout << idx << " " << par.totalMass() * intUnits.to_Msun << std::endl;
	////sampleParticles(bulge, 1e6 * intUnits.from_Msun, lower_pos.data(), upper_pos.data());
	//particles::writeSnapshot("model_thinDisk_test"+std::to_string(idx),
	//		par,
	//		"text", extUnits);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
for (int idx =0; idx < domain_data.size(); idx++) {
			auto row = domain_data[idx];
			std::vector<double> lower_pos = {row[0] * extUnits.lengthUnit, row[2] * extUnits.lengthUnit, row[4] * extUnits.lengthUnit};
			std::vector<double> upper_pos = {row[1] * extUnits.lengthUnit, row[3] * extUnits.lengthUnit, row[5] * extUnits.lengthUnit};
			SelectionFunctionLocalBox selectionFunction(lower_pos, upper_pos);
	
			std::cout << boxMass(model.components[0]->getDensity(), lower_pos.data(), upper_pos.data()) * intUnits.to_Msun << " ";
			std::cout << boxMass(model.components[1]->getDensity(), lower_pos.data(), upper_pos.data()) * intUnits.to_Msun << " ";
			std::cout << boxMass(model.components[2]->getDensity(), lower_pos.data(), upper_pos.data()) * intUnits.to_Msun << " ";
			std::cout << boxMass(model.components[3]->getDensity(), lower_pos.data(), upper_pos.data()) * intUnits.to_Msun << std::endl;
			
			galaxymodel::GalaxyModel bulge(*model.totalPotential, *model.actionFinder, *dfBulge);
			galaxymodel::GalaxyModel thinDisk(*model.totalPotential, *model.actionFinder, *dfThin);
			galaxymodel::GalaxyModel thickDisk(*model.totalPotential, *model.actionFinder, *dfThick);
			galaxymodel::GalaxyModel stellarHalo(*model.totalPotential, *model.actionFinder, *dfStellarHalo);
			galaxymodel::GalaxyModel dmHalo(*model.totalPotential, *model.actionFinder, *dfHalo);
			galaxymodel::GalaxyModel stellar(*model.totalPotential, *model.actionFinder, *dfStellar);
			PtrDensity ptrDensGasDisk = gasDisk.createDensity(model.totalPotential);
	
			particles::ParticleArrayCar par = sampleParticles(stellar, 1e5 * intUnits.from_Msun, lower_pos.data(), upper_pos.data());
			std::cout << idx << " " << par.totalMass() * intUnits.to_Msun << std::endl;
			//sampleParticles(bulge, 1e6 * intUnits.from_Msun, lower_pos.data(), upper_pos.data());
			particles::writeSnapshot("model_thinDisk_test"+std::to_string(idx),
					par,
					"text", extUnits);
		}


	return 0;
}
