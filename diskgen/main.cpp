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
#include "gas_scm.h"

#include "coord.h"
#include "galaxymodel_base.h"
#include "galaxymodel_selfconsistent.h"
#include "df_factory.h"
#include "math_base.h"
#include "particles_base.h"
#include "potential_base.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "potential_cylspline.h"
#include "potential_utils.h"
#include "particles_io.h"
#include "math_base.h"
#include "math_spline.h"
#include "units.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <time.h>

using potential::PtrDensity;
using potential::PtrPotential;

// define internal unit system - arbitrary numbers here! the result should not depend on their choice
const units::InternalUnits intUnits(1*units::Kpc, 977.7922216807891*units::Myr);

// define external unit system describing the data (including the parameters in INI file)
const units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 1.*units::kms, 1.*units::Msun);

// Particle mass
const double stellarParticleMass = 1e5 * intUnits.from_Msun;
const double gasParticleMass = 1e5 * intUnits.from_Msun;
const double dmParticleMass = 1e6 * intUnits.from_Msun;


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
	const potential::BaseDensity& compGasDisk  = *model.components[3]->getDensity();
	const potential::BaseDensity& compGasHalo  = *model.components[4]->getDensity();
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
		"Gas Disk total mass="  << (compGasDisk.totalMass()  * intUnits.to_Msun) << " Msun"
		", rho(Rsolar,z=0)="    << (compGasDisk.density(pt0) * intUnits.to_Msun_per_pc3) <<
		", rho(Rsolar,z=1kpc)=" << (compGasDisk.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
		"Gas Halo total mass="  << (compGasHalo.totalMass()  * intUnits.to_Msun) << " Msun"
		", rho(Rsolar,z=0)="    << (compGasHalo.density(pt0) * intUnits.to_Msun_per_pc3) <<
		", rho(Rsolar,z=1kpc)=" << (compGasHalo.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
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
						/*gridSizez*/ 50, /*zmin*/ 0.01, /*zmax*/ 10),
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


// compute gas density
void computeGasDensity(std::vector<double>& gasParticleDens, const particles::ParticleArrayCar particles, const galaxymodel::SelfConsistentModel& model) 
{
	const potential::BaseDensity& compGasDisk  = *model.components[3]->getDensity();
	const potential::BaseDensity& compGasHalo  = *model.components[4]->getDensity();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i=0; i<particles.size(); i++) {
		gasParticleDens[i] = compGasDisk.density(particles[i].first);
		gasParticleDens[i] += compGasHalo.density(particles[i].first);
	}
}


int main()
{
	//////////////////// READ INI FILE //////////////////// 
	
	const std::string iniFileName = "SCM.ini";
	//const std::string iniFileName = "SCM_small.ini";
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
	  iniGasParams     = ini.findSection("Gas parameters"),
		iniPotenGasHalo  = ini.findSection("Potential gas halo"),
		iniDFGasHalo     = ini.findSection("DF gas halo");
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


	//////////////////// INITIALIZE THE MODEL ////////////////////
	
	// initialize density profiles of various components
	std::vector<PtrDensity> densityStellarDisk(2);
	PtrDensity densityBulge    = potential::createDensity(iniPotenBulge,    extUnits);
	PtrDensity densityDarkHalo = potential::createDensity(iniPotenDarkHalo, extUnits);
	densityStellarDisk[0]      = potential::createDensity(iniPotenThinDisk, extUnits);
	densityStellarDisk[1]      = potential::createDensity(iniPotenThickDisk,extUnits);
	PtrDensity densityGasDisk  = potential::createDensity(iniPotenGasDisk,  extUnits);
	PtrDensity densityGasHalo  = potential::createDensity(iniPotenGasHalo,  extUnits);


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
	model.components.push_back(galaxymodel::PtrComponent(
				new galaxymodel::ComponentStatic(densityGasHalo, false)));

	// initialize total potential of the model (first guess)
	updateTotalPotential(model);
	printoutInfo(model, "init");

	std::cout << "\033[1;33m**** STARTING MODELLING ****\033[0m\nInitial masses of density components: "
		"Mdisk="  << (model.components[0]->getDensity()->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mbulge=" << (densityBulge   ->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mhalo="  << (densityDarkHalo->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mgasdisk="   << (densityGasDisk ->totalMass() * intUnits.to_Msun) << " Msun, "
		"Mgashalo="   << (densityGasHalo ->totalMass() * intUnits.to_Msun) << " Msun\n";

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
	df::PtrDistributionFunction dfGasHalo = df::createDistributionFunction(
			iniDFGasHalo, model.totalPotential.get(), /*density not needed*/NULL, extUnits);

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

	// DF for the gas halo
	model.components[4] = galaxymodel::PtrComponent(
			new galaxymodel::ComponentWithSpheroidalDF(dfGasHalo, potential::PtrDensity(),
				iniSCMHalo.getInt("lmaxAngularSph"),
				iniSCMHalo.getInt("mmaxAngularSph"),
				iniSCMHalo.getInt("sizeRadialSph"),
				iniSCMHalo.getDouble("rminSph") * extUnits.lengthUnit,
				iniSCMHalo.getDouble("rmaxSph") * extUnits.lengthUnit));


	//////////////////// PERFORM ITERATIONS ////////////////////
	for(int iteration=1; iteration<=6; iteration++){
		doIteration(model, gasDisk, iteration);
	}

	// final gas disk potential without CylSpline approximation
	model.components[3] = galaxymodel::PtrComponent(new galaxymodel::ComponentStatic(
				gasDisk.createDensity(model.totalPotential), true));

	std::cout << "\033[1;32m**** FINISHED MODELLING ****\033[0m\n";
	printoutInfo(model, "Final");
	std::cout << std::endl;


	//////////////////// SAMPLE PARTICLES ////////////////////

	// read in domain coordinates
	std::ifstream inputFile("DomainCoord.txt");
	std::vector<std::vector<double> > domain_data;
	std::string line;
	while (getline(inputFile, line)) {
		std::istringstream iss(line);
		std::vector<double> numbers;
		double num;

		for (int i = 0; i < 6; ++i) {
			if (iss >> num) numbers.push_back(num);
		}

		domain_data.push_back(numbers);
	}
	inputFile.close();

for (int idx =0; idx < domain_data.size(); idx++) {
	std::cout << "########## DOMAIN " << idx << " ##########" << std::endl;
	auto row = domain_data[idx];
	//for (int i = 0; i <6; i++) row[i] = math::clip(row[i], -300., 300.);
	std::vector<double> lower_pos = {row[0] * extUnits.lengthUnit, row[2] * extUnits.lengthUnit, row[4] * extUnits.lengthUnit};
	std::vector<double> upper_pos = {row[1] * extUnits.lengthUnit, row[3] * extUnits.lengthUnit, row[5] * extUnits.lengthUnit};

	double massStellarDiskHalo = boxMass(model.components[0]->getDensity(), lower_pos.data(), upper_pos.data());
	double massBulge = boxMass(model.components[1]->getDensity(), lower_pos.data(), upper_pos.data());
	double massDM = boxMass(model.components[2]->getDensity(), lower_pos.data(), upper_pos.data());

	//galaxymodel::GalaxyModel thinDisk(*model.totalPotential, *model.actionFinder, *dfThin);
	//galaxymodel::GalaxyModel thickDisk(*model.totalPotential, *model.actionFinder, *dfThick);
	//galaxymodel::GalaxyModel stellarHalo(*model.totalPotential, *model.actionFinder, *dfStellarHalo);
	galaxymodel::GalaxyModel stellar(*model.totalPotential, *model.actionFinder, *dfStellar);
	galaxymodel::GalaxyModel bulge(*model.totalPotential, *model.actionFinder, *dfBulge);
	galaxymodel::GalaxyModel dmHalo(*model.totalPotential, *model.actionFinder, *dfHalo);
	PtrDensity ptrDensGasDisk = gasDisk.createDensity(model.totalPotential);
	galaxymodel::GalaxyModel gasHalo(*model.totalPotential, *model.actionFinder, *dfGasHalo);

	particles::ParticleArrayCar thinDiskParticles, thickDiskParticles, stellarHaloParticles, stellarParticles, bulgeParticles, dmHaloParticles, gasDiskParticles, gasHaloParticles;

	// Sample stellar (thin disk, thick disk, and stellar halo) particles
	//thinDiskParticles = sampleParticles(thinDisk, stellarParticleMass, lower_pos.data(), upper_pos.data());
	//thickDiskParticles = sampleParticles(thickDisk, stellarParticleMass, lower_pos.data(), upper_pos.data());
	//stellarHaloParticles = sampleParticles(stellarHalo, stellarParticleMass, lower_pos.data(), upper_pos.data());
	stellarParticles = sampleParticles(stellar, stellarParticleMass, lower_pos.data(), upper_pos.data());
	std::cout << "  Stellar Disk + Stellar Halo: " << stellarParticles.size() << " particles (";
	std::cout << stellarParticles.totalMass() * intUnits.to_Msun << " Msun)" << std::endl;

	// Sample  bulge particles
	if(massBulge > stellarParticleMass) {
		bulgeParticles = sampleParticles(bulge, stellarParticleMass, lower_pos.data(), upper_pos.data());
	}
	std::cout << "  Bulge: " << bulgeParticles.size() << " particles (";
	std::cout << bulgeParticles.totalMass() * intUnits.to_Msun << " Msun)" << std::endl;

	// Sample DM halo particles
	dmHaloParticles = sampleParticles(dmHalo, dmParticleMass, lower_pos.data(), upper_pos.data());
	std::cout << "  DM Halo: " << dmHaloParticles.size() << " particles (";
	std::cout << dmHaloParticles.totalMass() * intUnits.to_Msun << " Msun)" << std::endl;

	// Sample gas disk particles
	gasDiskParticles = sampleParticles(ptrDensGasDisk, model.totalPotential, gasDisk, gasParticleMass, lower_pos.data(), upper_pos.data());
	std::vector<double> gasDiskParticleDens(gasDiskParticles.size());
	computeGasDensity(gasDiskParticleDens, gasDiskParticles, model);
	std::cout << "  Gas Disk: " << gasDiskParticles.size() << " particles (";
	std::cout << gasDiskParticles.totalMass() * intUnits.to_Msun << " Msun)" << std::endl;

	// Sample gas halo particles
	gasHaloParticles = sampleParticles(gasHalo, gasParticleMass, lower_pos.data(), upper_pos.data());
	std::vector<double> gasHaloParticleDens(gasHaloParticles.size());
	computeGasDensity(gasHaloParticleDens, gasHaloParticles, model);
	std::cout << "  Gas Halo: " << gasHaloParticles.size() << " particles (";
	std::cout << gasHaloParticles.totalMass() * intUnits.to_Msun << " Msun)" << std::endl;

	std::cout << " Total: " << stellarParticles.size() + bulgeParticles.size() + dmHaloParticles.size() + gasDiskParticles.size() + gasHaloParticles.size() << " particles ";
	std::cout << std::endl;

	// How to access the particle data
	//std::cout << std::left 
	//	<< std::setw(15) << "x [kpc]" 
	//	<< std::setw(15) << "y [kpc]"
	//	<< std::setw(15) << "z [kpc]"
	//	<< std::setw(15) << "vx [km/s]"
	//	<< std::setw(15) << "vy [km/s]"
	//	<< std::setw(15) << "vz [km/s]"
	//	<< std::setw(15) << "m [Msun]" << std::endl;
	//for (size_t i = 0; i < 10; i++) {
	//std::cout << std::left
	//	<< std::setw(15) << stellarParticles[i].first.x / extUnits.lengthUnit
	//	<< std::setw(15) << stellarParticles[i].first.y / extUnits.lengthUnit
	//	<< std::setw(15) << stellarParticles[i].first.z / extUnits.lengthUnit
	//	<< std::setw(15) << stellarParticles[i].first.vx / extUnits.velocityUnit
	//	<< std::setw(15) << stellarParticles[i].first.vy / extUnits.velocityUnit
	//	<< std::setw(15) << stellarParticles[i].first.vz / extUnits.velocityUnit
	//	<< std::setw(15) << stellarParticles[i].second / extUnits.massUnit << std::endl;
	//}


	// Write particles to file in ascii format if needed
	//particles::writeSnapshot("model_stellar_"+std::to_string(idx), stellarParticles, "text", extUnits);
	//particles::writeSnapshot("model_bulge_"+std::to_string(idx), bulgeParticles, "text", extUnits);
	//particles::writeSnapshot("model_dmHalo_"+std::to_string(idx), dmHaloParticles, "text", extUnits);
	//particles::writeSnapshot("model_gasDisk_"+std::to_string(idx), gasDiskParticles, "text", extUnits);
	//particles::writeSnapshot("model_gasHalo_"+std::to_string(idx), gasHaloParticles, "text", extUnits);
}

return 0;
}
