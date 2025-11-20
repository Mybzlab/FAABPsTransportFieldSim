# FAABPsTransportFieldSim

Simulation environment of a Force-Aligning Active Brownian Particle (FAABP) swarm with walls and a passive payload.

Hyperparameters are defined at the top of main.py, where you can also run the whole simulation from. 

Use run_tests.py to run unit tests for every function + an integration test. This is useful for verifying that everything is working correctly while changing the simulation. It is still good, though, to run and view the mp4 of the simulation, as a visual confirmation that everything is behaving as expected!

TODO:
- Precompute force fields for walls.
- Walls with curvature.