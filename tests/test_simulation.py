import pytest
import numpy as np
from src.simulation import (
    compute_all_forces,
    update_orientation_vectors
)


class TestOrientationUpdate:
    """Tests for orientation vector updates."""

    def test_update_orientation_vectors_no_noise(self):
        """Test orientation update without noise."""
        orientations = np.array([[1.0, 0.0]])
        forces = np.array([[0.0, 1.0]])  # Force in y direction
        curvity = np.array([1.0])
        dt = 0.01
        rot_diffusion = np.array([0.0])  # No noise
        n_particles = 1

        new_orientations = update_orientation_vectors(
            orientations, forces, curvity, dt, rot_diffusion, n_particles
        )

        # Should rotate based on torque
        # Torque = curvity * (n × F) = 1.0 * (1*1 - 0*0) = 1.0
        # Rotation should be counterclockwise
        assert new_orientations[0, 1] > 0  # Should have positive y component
        # Should be normalized
        assert abs(np.linalg.norm(new_orientations[0]) - 1.0) < 1e-6

    def test_update_orientation_vectors_no_force(self):
        """Test orientation update with no force."""
        orientations = np.array([[1.0, 0.0]])
        forces = np.array([[0.0, 0.0]])
        curvity = np.array([1.0])
        dt = 0.01
        rot_diffusion = np.array([0.0])
        n_particles = 1

        new_orientations = update_orientation_vectors(
            orientations, forces, curvity, dt, rot_diffusion, n_particles
        )

        # No force, orientation should remain the same
        np.testing.assert_array_almost_equal(new_orientations, orientations)

    def test_update_orientation_vectors_normalization(self):
        """Test that orientations are always normalized."""
        np.random.seed(42)
        orientations = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        forces = np.random.randn(3, 2)
        curvity = np.random.randn(3)
        dt = 0.01
        rot_diffusion = np.array([0.1, 0.1, 0.1])
        n_particles = 3

        new_orientations = update_orientation_vectors(
            orientations, forces, curvity, dt, rot_diffusion, n_particles
        )

        # All orientations should be normalized
        for i in range(n_particles):
            norm = np.linalg.norm(new_orientations[i])
            assert abs(norm - 1.0) < 1e-6


class TestAllForces:
    """Tests for complete force computation."""

    def test_compute_all_forces_no_overlap(self):
        """Test force computation when no particles overlap."""
        positions = np.array([
            [10.0, 10.0],
            [20.0, 20.0]
        ])
        payload_pos = np.array([50.0, 50.0])
        radii = np.array([1.0, 1.0])
        payload_radius = 5.0
        stiffness = 10.0
        n_particles = 2
        box_size = 100.0
        walls = np.zeros((0, 4))

        particle_forces, payload_force = compute_all_forces(
            positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
        )

        # No overlap means no forces
        np.testing.assert_array_almost_equal(particle_forces, np.zeros((2, 2)))
        np.testing.assert_array_almost_equal(payload_force, np.zeros(2))

    def test_compute_all_forces_with_overlap(self):
        """Test force computation when particles overlap."""
        positions = np.array([
            [10.0, 10.0],
            [11.5, 10.0]  # Overlapping with first particle
        ])
        payload_pos = np.array([50.0, 50.0])
        radii = np.array([1.0, 1.0])
        payload_radius = 5.0
        stiffness = 10.0
        n_particles = 2
        box_size = 100.0
        walls = np.zeros((0, 4))

        particle_forces, payload_force = compute_all_forces(
            positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
        )

        # Particles overlap, should have repulsive forces
        # Particle 0 should be pushed left (negative x)
        assert particle_forces[0, 0] < 0
        # Particle 1 should be pushed right (positive x)
        assert particle_forces[1, 0] > 0
        # Forces should be equal and opposite (Newton's third law)
        np.testing.assert_array_almost_equal(particle_forces[0], -particle_forces[1])

    def test_compute_all_forces_payload_overlap(self):
        """Test forces when particle overlaps with payload."""
        positions = np.array([[50.0, 50.0]])
        payload_pos = np.array([52.0, 50.0])  # Overlapping
        radii = np.array([1.5])
        payload_radius = 1.5
        stiffness = 10.0
        n_particles = 1
        box_size = 100.0
        walls = np.zeros((0, 4))

        particle_forces, payload_force = compute_all_forces(
            positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
        )

        # Particle should be pushed left, payload right
        assert particle_forces[0, 0] < 0
        assert payload_force[0] > 0
        # Forces should be equal and opposite
        np.testing.assert_array_almost_equal(particle_forces[0], -payload_force)
