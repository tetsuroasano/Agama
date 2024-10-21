#include "gas_scm.h"

#include "actions_base.h"
#include "math_sample.h"
#include "particles_base.h"
#include "potential_utils.h"
#include "math_core.h"
#include <iostream>
#include <cmath>

GasDisk::GasDisk(double S0, double R0, double temp, double mu, double kB_over_mp) :
			S0(S0), R0(R0), temp(temp), mu(mu), kB_over_mp(kB_over_mp), c2(kB_over_mp*temp/mu) {
				int_energy = c2;
				if (gamma != 1.0) {
					int_energy /= (gamma - 1.0);
				}
			}

/** calcuate density */
double GasDisk::density(const coord::PosCyl &pos, const PtrPotential& totalPot) const {
	// TODO : automatically set the upper limit of the integral
	double I = math::integrate(ExpPhic2(pos, totalPot, c2), 0, 10, 1e-3);
	double rho0 = 0.5 * S0 * exp(-pos.R / R0) / I;
	return rho0 * exp(- Phi_z(pos, totalPot)/ c2);
}

/// Phi_z(R, z) := Phi(R, z) - Phi(R,0) used in the density calculation
double GasDisk::Phi_z(const coord::PosCyl &pos, const PtrPotential& totalPot) {
	double pot_value = totalPot->value(pos);
	pot_value -= totalPot->value(coord::PosCyl(pos.R, 0, 0));
	return pot_value;
}

///Helper class for calculation of the density
GasDisk::ExpPhic2::ExpPhic2(const coord::PosCyl &pos, const PtrPotential& totalPot, double c2): pos(pos), totalPot(totalPot), c2(c2){};

double GasDisk::ExpPhic2::value(const double z) const {
			return exp(- Phi_z(coord::PosCyl(pos.R, z, 0), totalPot) / c2);
		}


void GasDisk::computeVelocity(coord::PosVelCyl& point, const PtrPotential& totalPot) const {
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
void GasDisk::computeVelocity(coord::PosVelCar& point, const PtrPotential& totalPot) const {
	coord::PosVelCyl cyl = coord::toPosVelCyl(point);
	computeVelocity(cyl, totalPot);
	point = coord::toPosVelCar(cyl);
}
void GasDisk::computeVelocity(coord::PosVelSph& point, const PtrPotential& totalPot) const {
	coord::PosVelCyl cyl = coord::toPosVelCyl(point);
	computeVelocity(cyl, totalPot);
	point = coord::toPosVelSph(cyl);
}


// IsothermalGasDisk density pointer
GasDisk::IsothermalGasDisk::IsothermalGasDisk(GasDisk* gasDisk,
		const PtrPotential& totalPot):
	gasDisk(gasDisk), totalPot(totalPot){};

double GasDisk::IsothermalGasDisk::densityCyl(const coord::PosCyl &pos, double time) const { 
	return gasDisk->density(pos, totalPot);
}
double GasDisk::IsothermalGasDisk::densityCar(const coord::PosCar &pos, double time) const { 
	return densityCyl(toPosCyl(pos), time);
}
double GasDisk::IsothermalGasDisk::densitySph(const coord::PosSph &pos, double time) const {
	return densityCyl(toPosCyl(pos), time);
}

PtrDensity GasDisk::createDensity(const PtrPotential& totalPot) {
	return PtrDensity(new IsothermalGasDisk(this, totalPot));
}


/// Helper class for computing the density at a given point in Cartesian coordinates
DensityIntegrandCar::DensityIntegrandCar(const potential::BaseDensity& _dens) : dens(_dens) {}

void DensityIntegrandCar::eval(const double vars[], double values[]) const {
	coord::PosCar pos(vars[0], vars[1], vars[2]);
	values[0] = dens.density(pos, /*time*/0);
	if (!isFinite(values[0]) || values[0] < 0) values[0] = 0;
}


/// Helper class for sampling position and velocity from a DF (not scaled)
DFIntegrandCar::DFIntegrandCar(const galaxymodel::GalaxyModel& _model): model(_model){}

void DFIntegrandCar::eval(const double vars[], double values[]) const {
			actions::Actions act = model.actFinder.actions(
					coord::toPosVelCyl(coord::PosVelCar(vars[0], vars[1], vars[2], vars[3], vars[4], vars[5]))
					);
			values[0] = model.distrFunc.value(act);
			if (!isFinite(values[0]) || values[0] < 0) values[0] = 0;
		}



/// Box selection function (not used now)
SelectionFunctionLocalBox::SelectionFunctionLocalBox(const std::vector<double>& lower_pos, const std::vector<double>& upper_pos):
			lower_pos(lower_pos), upper_pos(upper_pos) {};
double SelectionFunctionLocalBox::value(const coord::PosVelCar& point) const { 
			if (contains(point)) {return 1.;}
			else {return 0.;}
		}
bool SelectionFunctionLocalBox::contains(const coord::PosVelCar &point) const {
	return point.x >= lower_pos[0] && point.x <= upper_pos[0] &&
		point.y >= lower_pos[1] && point.y <= upper_pos[1] &&
		point.z >= lower_pos[2] && point.z <= upper_pos[2];
}



/// compute the mass of a density component within a given box
double boxMass(const PtrDensity ptrDens, const double lower[], const double upper[]) {
	double mass, massErr;
	int numEval;
	math::integrateNdim(DensityIntegrandCar(*ptrDens),
			lower, upper, 1e-4, 1e6, &mass, &massErr, &numEval);

	return mass;
}


/// sample position from a density distribution in Cartesian coordinates
math::Matrix<double> sampleDensityCar(const PtrDensity ptrDens, const size_t numPoints,
		const double lower[], const double upper[], double* totalMass, double* errorMass) {
	math::Matrix<double> result;

	try{
		math::sampleNdim(DensityIntegrandCar(*ptrDens), lower, upper, numPoints, result, NULL, totalMass, errorMass);
	} catch (const std::runtime_error& e) {
		if (std::string(e.what()) == "sampleNdim: function is identically zero inside the region" ||
				std::string(e.what()) == "Error in sampleNdim: refinement procedure did not converge"
			 ) {
			std::cerr << "Caught specific runtime error: " << e.what() << std::endl;
			return math::Matrix<double>();
		} else {
			std::cerr << "Unexpected runtime error: " << e.what() << std::endl;
			std::exit(EXIT_FAILURE); 
		}
	} catch (const std::exception& e) {
		std::cerr << "Unexpected exception: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);  
	}
	return result;
};


/// sample particles from a DF within a given box
math::Matrix<double> samplePosVelCar(const galaxymodel::GalaxyModel model, const size_t numPoints, const double lower_pos[], const double upper_pos[], double* totalMass, double* errorMass) {
	double Phi;
	model.potential.eval(coord::PosCar(0., 0., 0.), &Phi);
	double vesc = sqrt(-2.*Phi);
	double lower[6] = {lower_pos[0], lower_pos[1], lower_pos[2], -vesc*0.57, -vesc*0.57, -vesc*0.57};
	double upper[6] = {upper_pos[0], upper_pos[1], upper_pos[2],  vesc*0.57,  vesc*0.57,  vesc*0.57};
	math::Matrix<double> result;

	try {
		math::sampleNdim(DFIntegrandCar(model), lower, upper, numPoints, result, NULL, totalMass, errorMass);
	} catch (const std::runtime_error& e) {
		if (std::string(e.what()) == "sampleNdim: function is identically zero inside the region" ||
				std::string(e.what()) == "Error in sampleNdim: refinement procedure did not converge"
			 ) {
			std::cerr << "Caught specific runtime error: " << e.what() << std::endl;
			return math::Matrix<double>();
		} else {
			std::cerr << "Unexpected runtime error: " << e.what() << std::endl;
			std::exit(EXIT_FAILURE); 
		}
	} catch (const std::exception& e) {
		std::cerr << "Unexpected exception: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);  
	}

	return result;
};


/// sample particles from self-consistent model 
particles::ParticleArrayCar sampleParticles(
		const galaxymodel::GalaxyModel model, const double partMass,
		const double lower_pos[], const double upper_pos[]) {
	double totalMass, errorMass;
	math::Matrix<double> result;
	particles::ParticleArrayCar	points;

	// Sample 1M points to estimate the total mass
	result =  samplePosVelCar(model, (int)1e6, lower_pos, upper_pos,  &totalMass, &errorMass);

	if (totalMass < partMass) { return particles::ParticleArrayCar(); }

	// If we need more points, sample again
	size_t numPoints = (size_t)(totalMass / partMass);
	if (numPoints > result.rows()) {
		result =  samplePosVelCar(model, numPoints, lower_pos, upper_pos,
				                      &totalMass, &errorMass);
	}

	for(size_t i=0; i<numPoints; i++) {
		points.add(coord::PosVelCar(result(i,0), result(i,1), result(i,2),
																result(i,3), result(i,4), result(i,5)),
				                        partMass);
	}

	return points;
}


/// sample particles from the gas disk density component and assign velocities
particles::ParticleArrayCar sampleParticles(
		const PtrDensity& ptrDens, const PtrPotential& totalPot, const GasDisk& gasDisk,
		const double partMass,
		const double lower_pos[], const double upper_pos[]) {
	double totalMass, errorMass;
	math::Matrix<double> result;
	particles::ParticleArrayCar	points;
	 
	// Sample 1M points to estimate the total mass
	result =  sampleDensityCar(ptrDens, (int)1e6, lower_pos, upper_pos,  &totalMass, &errorMass);

	if (totalMass < partMass) { return particles::ParticleArrayCar(); }

	// If we need more points, sample again
	size_t numPoints = (size_t)(totalMass / partMass);
	if (numPoints > result.rows()) {
		result =  sampleDensityCar(ptrDens, numPoints, lower_pos, upper_pos,
				                       &totalMass, &errorMass);
	}

	for(size_t i=0; i<numPoints; i++) {
		points.add(coord::PosVelCar(result(i,0), result(i,1), result(i,2), 0., 0., 0.),
				       partMass);
	}
	for(size_t i=0; i<numPoints; i++) {
		gasDisk.computeVelocity(points[i].first, totalPot);
	}

	return points;
}

