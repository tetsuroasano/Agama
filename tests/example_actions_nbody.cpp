/** \file    example_actions_nbody.cpp
    \author  Eugene Vasiliev
    \date    2015-2016

    This example demonstrates the use of action finder (Staeckel approximation)
    to compute actions for particles from an N-body simulation.
    The N-body system consists of a disk and a halo,
    the two components being stored in separate text files.
    They are not provided in the distribution, but could be created by running
    example_self_consistent_model.exe
    The potential is computed from the snapshot itself, by creating a suitable
    potential expansion for each component: Multipole for the halo and CylSpline
    for the disk. This actually takes most of the time. We save the constructed
    potentials to text files, and on the subsequent launches of this program
    they are loaded from these files, speeding up the initialization.
    Then we compute actions for all particles from the disk component,
    and store them in a text file.

    An equivalent example in Python is located in pytests folder;
    it uses the same machinery through a Python extension of the C++ library.
*/
#include "potential_cylspline.h"
#include "potential_multipole.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include "units.h"
#include <iostream>
#include <fstream>
#include <ctime>

int main() {
    // #1. Set up units.
    // some arbitrary internal units (note: the end result should not depend on their choice)
    const units::InternalUnits unit(2.7183 * units::Kpc, 3.1416 * units::Myr);
    // input snapshot is generated by the example_self_consistent_model program,
    // and it provided in the standard galactic units (kpc, km/s, Msun)
    const units::ExternalUnits extUnits(unit, units::Kpc, units::kms, units::Msun);

    // #2. Get in N-body snapshots
    clock_t tbegin=std::clock();
    particles::ParticleArrayCar diskParticles, haloParticles;
    try {
        diskParticles = particles::readSnapshot("model_stars_final", extUnits);
        haloParticles = particles::readSnapshot("model_dm_final", extUnits);
    }
    catch(...) {
        std::cout << "Input snapshot files are not available; "
            "you may create them by running example_self_consistent_model.exe\n";
        return 0;
    }
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to load " <<
        diskParticles.size() << " disk particles (total mass=" <<
        diskParticles.totalMass()*unit.to_Msun << " Msun) and " <<
        haloParticles.size() << " halo particles (total mass=" <<
        haloParticles.totalMass()*unit.to_Msun << " Msun)\n";

    // #3. Initialize the gravitational potential
    potential::PtrPotential haloPot, diskPot;
    try{
        // #3a. Try to load the potentials from previously saved text files
        diskPot = potential::readPotential("model_stars_final.ini", extUnits);
        haloPot = potential::readPotential("model_dm_final.ini", extUnits);
    }
    catch(std::exception&) {
        // #3b. These files do not exist on the first run, so we need to
        // construct the potential approximations from these particles
        tbegin=std::clock();
        haloPot = potential::Multipole::create(
            haloParticles, coord::ST_AXISYMMETRIC, 2 /*lmax*/, 0 /*mmax*/, 20 /*gridSizeR*/);
        std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC <<
            " s to init " << haloPot->name() << " potential for the halo;  "
            "value at origin=" << haloPot->value(coord::PosCar(0,0,0)) * pow_2(unit.to_kms) << " (km/s)^2\n";

        tbegin=std::clock();
        // manually specify the spatial grid for the disk potential,
        // although one may rely on the automatic choice of these parameters (as we did for the halo)
        double Rmin = 0.2 *unit.from_Kpc, Rmax = 100*unit.from_Kpc,
               Zmin = 0.05*unit.from_Kpc, Zmax = 50*unit.from_Kpc;
        int gridSizeR = 20, gridSizeZ=20;
        diskPot = potential::CylSpline::create(
            diskParticles, coord::ST_AXISYMMETRIC, 0 /*mmax*/,
            gridSizeR, Rmin, Rmax, gridSizeZ, Zmin, Zmax);
        std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC <<
            " s to init " << diskPot->name() << " potential for the disk;  "
            "value at origin=" << diskPot->value(coord::PosCar(0,0,0)) * pow_2(unit.to_kms) << " (km/s)^2\n";

        // (optional) store the potential coefs into a file and next time load them back to speed up process
        writePotential("model_stars_final.ini", *diskPot, extUnits);
        writePotential("model_dm_final.ini", *haloPot, extUnits);
    }

    // #3c. Combine the two components
    std::vector<potential::PtrPotential> components(2);
    components[0] = haloPot;
    components[1] = diskPot;
    potential::PtrPotential totalPot(new potential::Composite(components));

    // #4. Compute actions
    tbegin=std::clock();
    actions::ActionFinderAxisymFudge actFinder(totalPot);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init action finder\n";
    tbegin=std::clock();
    int nbody = diskParticles.size();
    std::vector<actions::Actions> acts(nbody);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1024)
#endif
    for(int i=0; i<nbody; i++) {
        acts[i] = actFinder.actions(toPosVelCyl(diskParticles[i].first));
    }
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to compute actions  ("<<
        diskParticles.size() * 1.0*CLOCKS_PER_SEC / (std::clock()-tbegin) << " actions per second)\n";

    // #5. Store results
    std::ofstream strm("disk_actions.txt");
    strm << "# R[Kpc]\tz[Kpc]\tJ_r[Kpc*km/s]\tJ_z[Kpc*km/s]\tJ_phi[Kpc*km/s]\tE[(km/s)^2]\n";
    for(int i=0; i<nbody; i++) {
        const coord::PosVelCyl point = toPosVelCyl(diskParticles[i].first);
        strm <<
            point.R*unit.to_Kpc << "\t" <<
            point.z*unit.to_Kpc << "\t" <<
            acts[i].Jr*unit.to_Kpc_kms << "\t" <<
            acts[i].Jz*unit.to_Kpc_kms << "\t" <<
            acts[i].Jphi*unit.to_Kpc_kms << "\t" <<
            totalEnergy(*totalPot, point)*pow_2(unit.to_kms) << "\n";
    }
}
