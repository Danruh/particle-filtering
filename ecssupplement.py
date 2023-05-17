import numpy as np
from datetime import datetime, timedelta
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from scipy.stats import multivariate_normal

np.random.seed(1991)

def generate_linearish_path(q_x, q_y) -> GroundTruthPath:
    start_time = datetime.now()

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x), ConstantVelocity(q_y)])

    # Generate the truth path
    truth_path = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

    num_steps = 20
    for k in range(1, num_steps + 1):
        truth_path.append(GroundTruthState(
            transition_model.function(truth_path[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k)))

    return truth_path, transition_model, start_time


def linear_measurement(truth_path, w=5) -> list:
    R = np.array([[1, 0],  # Covariance matrix for Gaussian PDF
                  [0, 1]]) * w

    measurement_model = LinearGaussian(
        ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2),  # Mapping measurement vector index to state index
        noise_covar=R
        )
    
    # Generate the measurements
    measurements = []
    for state in truth_path:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))
        
    return measurements, measurement_model


def bearing_range_measurement(truth_path) -> list:
    sensor_x = 50  # Placing the sensor off-centre
    sensor_y = 0

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
        # bearing and 1 metre in range
        translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
        # sensor in cartesian.
    )

    measurements = []
    for state in truth_path:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement, timestamp=state.timestamp,
                                      measurement_model=measurement_model))
        
    return measurements, measurement_model


def linear_kalman_filter(transition_model, measurement_model, measurements, start_time) -> Track:
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Create initial conditions (first prior)
    prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

    return track


def extended_kalman_filter(transition_model, measurement_model, measurements, start_time) -> Track:
    predictor = ExtendedKalmanPredictor(transition_model)    
    updater = ExtendedKalmanUpdater(measurement_model)

    prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

    return track


def gaussian_entropy(state) -> list:
    mean  = state.state_vector.flatten()
    covar = state.covar

    gaussian = multivariate_normal(mean=mean, cov=covar)

    return gaussian.entropy()


def kl_divergence(sensor_0, sensor_1):
    ...


def chernoff_divergence():
    ...


def renyi_divergence():
    ...


if __name__ == '__main__':
    print(f"Name: {__name__}")
    print("This is a supplementary module for this exercise")