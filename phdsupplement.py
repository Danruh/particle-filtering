####################################################
# Supplementary code for the phd_filtering notebook
# Code modified from https://stonesoup.readthedocs.io/en/v0.1b12/auto_tutorials/filters/GMPHDTutorial.html#sphx-glr-auto-tutorials-filters-gmphdtutorial-py
####################################################

from matplotlib import pyplot as plt
import numpy as np
from ordered_set import OrderedSet
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.plotter import Plotterly

from stonesoup.models.measurement.linear import LinearGaussian

from scipy.stats import uniform

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter

from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.updater.pointprocess import PHDUpdater

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser

from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer

from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import CovarianceMatrix

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def gen_ground_truths(start_time, birth_probability, death_probability, number_steps, q=0.3):
    truths_by_time = []
    transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(q), ConstantVelocity(q)))

    truths = OrderedSet()
    current_truths = set()
    start_truths = set()

    truths_by_time.append([])
    for i in range(3):
        x, y = initial_position = np.random.uniform(-30, 30, 2)  # Range [-30, 30] for x and y
        x_vel, y_vel = (np.random.rand(2))*2 - 1  # Range [-1, 1] for x and y velocity
        state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time)

        truth = GroundTruthPath([state])
        current_truths.add(truth)
        truths.add(truth)
        start_truths.add(truth)
        truths_by_time[0].append(state)

    # Simulate the ground truth over time
    for k in range(number_steps):
        truths_by_time.append([])
        # Death
        for truth in current_truths.copy():
            if np.random.rand() <= death_probability:
                current_truths.remove(truth)
        # Update truths
        for truth in current_truths:
            updated_state = GroundTruthState(
                transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=start_time + timedelta(seconds=k))
            truth.append(updated_state)
            truths_by_time[k].append(updated_state)
        # Birth
        for _ in range(np.random.poisson(birth_probability)):
            x, y = initial_position = np.random.rand(2) * [120, 120]  # Range [0, 20] for x and y
            x_vel, y_vel = (np.random.rand(2))*2 - 1  # Range [-1, 1] for x and y velocity
            state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time + timedelta(seconds=k))

            # Add to truth set for current and for all timestamps
            truth = GroundTruthPath([state])
            current_truths.add(truth)
            truths.add(truth)
            truths_by_time[k].append(state)

    return truths, truths_by_time, start_truths, transition_model



def gen_detectors(truths, start_time, number_steps, w=0.75, probability_detection=0.9, clutter_rate=3.0):
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[1, 0],
                              [0, 1]]) * w
        )
    
    all_measurements = []
    # The probability detection and clutter rate play key roles in the posterior intensity.
    # They can be changed to see their effect.

    for k in range(number_steps):
        measurement_set = set()
        timestamp = start_time + timedelta(seconds=k)

        for truth in truths:
            try:
                truth_state = truth[timestamp]
            except IndexError:
                continue

            if np.random.rand() <= probability_detection:
                # Generate actual detection from the state
                measurement = measurement_model.function(truth_state, noise=True)
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                  groundtruth_path=truth,
                                                  timestamp=truth_state.timestamp,
                                                  measurement_model=measurement_model))

        # Generate clutter at this time-step
        for _ in range(np.random.poisson(clutter_rate)):
            x = uniform.rvs(-200, 400)
            y = uniform.rvs(-200, 400)
            measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=timestamp,
                                        measurement_model=measurement_model))

        all_measurements.append(measurement_set)

    return all_measurements, measurement_model


def run_tracker(all_measurements, measurement_model, transition_model, probability_detection, death_probability, clutter_rate, start_truths, start_time):
    kalman_updater = KalmanUpdater(measurement_model)

    # Area in which we look for target. Note that if a target appears outside of this area the
    # filter will not pick up on it.
    meas_range = np.array([[-1, 1], [-1, 1]])*200
    clutter_spatial_density = clutter_rate/np.prod(np.diff(meas_range))

    from stonesoup.updater.pointprocess import PHDUpdater
    updater = PHDUpdater(
        kalman_updater,
        clutter_spatial_density=clutter_spatial_density,
        prob_detection=probability_detection,
        prob_survival=1-death_probability)
    
    # Generate predictors
    kalman_predictor = KalmanPredictor(transition_model)
    base_hypothesiser = DistanceHypothesiser(kalman_predictor, kalman_updater, Mahalanobis(), missed_distance=3)

    hypothesiser = GaussianMixtureHypothesiser(base_hypothesiser, order_by_detection=True)

    # Initialise Gaussian reducer
    merge_threshold = 5
    prune_threshold = 1e-8

    reducer = GaussianMixtureReducer(
        prune_threshold=prune_threshold,
        pruning=True,
        merge_threshold=merge_threshold,
        merging=True
    )

    covar = CovarianceMatrix(np.diag([10, 5, 10, 5]))

    tracks = set()
    for truth in start_truths:
        new_track = TaggedWeightedGaussianState(
                state_vector=truth.state_vector,
                covar=covar**2,
                weight=0.25,
                tag=TaggedWeightedGaussianState.BIRTH,
                timestamp=start_time)
        tracks.add(Track(new_track))

    reduced_states = set([track[-1] for track in tracks])

    birth_covar = CovarianceMatrix(np.diag([1000, 2, 1000, 2]))
    birth_component = TaggedWeightedGaussianState(
        state_vector=[0, 0, 0, 0],
        covar=birth_covar**2,
        weight=0.25,
        tag='birth',
        timestamp=start_time
    )

    # Run the tracker!
    all_gaussians = []
    tracks_by_time = []

    state_threshold = 0.25

    for n, measurements in enumerate(all_measurements):
        tracks_by_time.append([])
        all_gaussians.append([])

        # The hypothesiser takes in the current state of the Gaussian mixture. This is equal to the list of
        # reduced states from the previous iteration. If this is the first iteration, then we use the priors
        # defined above.
        current_state = reduced_states

        # At every time step we must add the birth component to the current state
        if measurements:
            time = list(measurements)[0].timestamp
        else:
            time = start_time + timedelta(seconds=n)
        birth_component.timestamp = time
        current_state.add(birth_component)

        # Generate the set of hypotheses
        hypotheses = hypothesiser.hypothesise(current_state,
                                              measurements,
                                              timestamp=time,
                                              # keep our hypotheses ordered by detection, not by track
                                              order_by_detection=True)

        # Turn the hypotheses into a GaussianMixture object holding a list of states
        updated_states = updater.update(hypotheses)

        # Prune and merge the updated states into a list of reduced states
        reduced_states = set(reducer.reduce(updated_states))

        # Add the reduced states to the track list. Each reduced state has a unique tag. If this tag matches the tag of a
        # state from a live track, we add the state to that track. Otherwise, we generate a new track if the reduced
        # state's weight is high enough (i.e. we are sufficiently certain that it is a new track).
        for reduced_state in reduced_states:
            # Add the reduced state to the list of Gaussians that we will plot later. Have a low threshold to eliminate some
            # clutter that would make the graph busy and hard to understand
            if reduced_state.weight > 0.05: all_gaussians[n].append(reduced_state)

            tag = reduced_state.tag
            # Here we check to see if the state has a sufficiently high weight to consider being added.
            if reduced_state.weight > state_threshold:
                # Check if the reduced state belongs to a live track
                for track in tracks:
                    track_tags = [state.tag for state in track.states]

                    if tag in track_tags:
                        track.append(reduced_state)
                        tracks_by_time[n].append(reduced_state)
                        break
                else:  # Execute if no "break" is hit; i.e. no track with matching tag
                    # Make a new track out of the reduced state
                    new_track = Track(reduced_state)
                    tracks.add(new_track)
                    tracks_by_time[n].append(reduced_state)

    return tracks
    

def get_mixture_density(x, y, weights, means, sigmas):
    # We use the quantiles as a parameter in the multivariate_normal function. We don't need to pass in any quantiles,
    # but the last axis must have the components x and y
    quantiles = np.empty(x.shape + (2,))  # if  x.shape is (m,n) then quantiles.shape is (m,n,2)
    quantiles[:, :, 0] = x
    quantiles[:, :, 1] = y

    # Go through each gaussian in the list and add its PDF to the mixture
    z = np.zeros(x.shape)
    for gaussian in range(len(weights)):
        z += weights[gaussian]*multivariate_normal.pdf(x=quantiles, mean=means[gaussian, :], cov=sigmas[gaussian])
    return z