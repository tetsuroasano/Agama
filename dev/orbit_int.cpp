#include "orbit.h"
#include "potential_factory.h"
#include "particles_io.h"
#include "potential_utils.h"
#include "units.h"
#include "utils.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

void displayProgress(int progressCount, int total, const std::chrono::time_point<std::chrono::high_resolution_clock>& startTime) {
    float progress = static_cast<float>(progressCount) / total;

    std::cout << "\rProgress: [";
    int barWidth = 50;  
    int pos = static_cast<int>(barWidth * progress);

    for (int j = 0; j < barWidth; ++j) {
        if (j < pos)
            std::cout << "=";
        else if (j == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }

    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%";

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - startTime;

    std::cout << " | Elapsed Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s";

    std::cout.flush();  
}

int main() {
	const units::InternalUnits unit(units::Kpc, 9.777922216807891 * units::Myr);
	const units::ExternalUnits extUnits(unit, units::Kpc, units::kms, 232508.54 * units::Msun);

	potential::PtrPotential totalPot;
	totalPot = potential::readPotential("../data/test.ini", extUnits);

	particles::ParticleArrayCar diskParticles;

	diskParticles = particles::readSnapshot("../data/snap_small", extUnits);
	int nbody = diskParticles.size();

	orbit::OrbitIntParams params(/*accuracy*/ 1e-8, /*maxNumSteps*/10000);
	double init_time = 0.;
	double total_time = 1.4;
	double timestep = 0.1;

	std::vector<orbit::Trajectory> trajectories(nbody);	

	auto startTime = std::chrono::high_resolution_clock::now();


#pragma omp parallel for schedule(dynamic,1024)
	for (int i = 0; i < nbody; i++){
		orbit::Trajectory traj;
		orbit::OrbitIntegrator<coord::Car> orbint(*totalPot, /*Omega*/0, params);
		orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(
					new orbit::RuntimeTrajectory(
						orbint, timestep, /*output*/ traj)));
		orbint.init(diskParticles[i].first, init_time);
		orbint.run(total_time);
		trajectories[i] = traj;

#pragma omp critical
		{
			displayProgress(i+1, nbody, startTime); 
		}

	}
	std::cout << std::endl;

  for (int t=0; t < trajectories[0].size(); t++) {
		particles::ParticleArrayCar snapshot;
		for (int i = 0; i < nbody; i++){
			snapshot.add(trajectories[i][t].first, diskParticles[i].second);
		}
		std::stringstream ss;
    ss << "snapshot_" << std::setw(5) << std::setfill('0') << t;
		particles::writeSnapshot(ss.str(), snapshot, "Gadget", extUnits);

	}

	return 0;
}
