/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static constexpr int kNumOfParticles{100};
static constexpr double kParticleInitialWeight{1.0};
static constexpr double kEpsilon{0.00001};
static constexpr double kZeroMean{0};

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  if (!initialized()) {

    num_particles = kNumOfParticles;

    const auto std_dev_x{std[0]};
    const auto std_dev_y{std[1]};
    const auto std_dev_theta{std[2]};

    for (int index{0}; index < num_particles; ++index) {
      Particle particle{};
      particle.id = index;
      particle.weight = kParticleInitialWeight;
      particle.x = get_gaussian_noise(x, std_dev_x);
      particle.y = get_gaussian_noise(y, std_dev_y);
      particle.theta = get_gaussian_noise(theta, std_dev_theta);
      particles.emplace_back(particle);
    }

    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  std::default_random_engine random_generator_engine{};
  std::normal_distribution<double> gaussian_distribution_x(0, std_pos[0]);
  std::normal_distribution<double> gaussian_distribution_y(0, std_pos[1]);
  std::normal_distribution<double> gaussian_distribution_theta(0, std_pos[2]);

  for (auto &particle : particles) {
    if (fabs(yaw_rate) < 0.00001) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      particle.x +=
          velocity / yaw_rate *
          (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y +=
          velocity / yaw_rate *
          (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }

    particle.x += gaussian_distribution_x(random_generator_engine);
    particle.y += gaussian_distribution_y(random_generator_engine);
    particle.theta += gaussian_distribution_theta(random_generator_engine);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {

  for (auto &observation : observations) {
    double minimum_distance{std::numeric_limits<double>::max()};
    int nearest_predicted_object{-1};
    for (const auto &prediction : predicted) {
      const auto distance{distance_between_landmarks(observation, prediction)};
      if (distance < minimum_distance) {
        minimum_distance = distance;
        nearest_predicted_object = prediction.id;
      }
    }
    observation.id = nearest_predicted_object;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  for (auto &particle : particles) {
    const double &particle_x{particle.x};
    const double &particle_y{particle.y};
    const auto &particle_theta{particle.theta};

    vector<LandmarkObs> predictions{};

    for (const auto &map_landmark : map_landmarks.landmark_list) {

      const float &map_landmark_x{map_landmark.x_f};
      const float &map_landmark_y{map_landmark.y_f};
      const int &map_landmark_id{map_landmark.id_i};

      if (std::fabs(map_landmark_x - particle_x) <= sensor_range &&
          std::fabs(map_landmark_y - particle_y) <= sensor_range) {
        predictions.emplace_back(
            LandmarkObs{map_landmark_id, map_landmark_x, map_landmark_y});
      }
    }

    std::vector<LandmarkObs> translated_observations{};
    for (const auto &observation : observations) {
      LandmarkObs translated_observation{};
      translated_observation.id = observation.id;
      translated_observation.x = std::cos(particle_theta) * observation.x -
                                 std::sin(particle_theta) * observation.y +
                                 particle_x;
      translated_observation.y = std::sin(particle_theta) * observation.x +
                                 std::cos(particle_theta) * observation.y +
                                 particle_y;
      translated_observations.emplace_back(translated_observation);
    }

    dataAssociation(predictions, translated_observations);
    particle.weight = 1.0;

    const auto observation_std_dev_x{std_landmark[0]};
    const auto observation_std_dev_y{std_landmark[1]};

    for (const auto &translated_observation : translated_observations) {
      double observed_x{}, observed_y{}, predicted_x, predicted_y;
      observed_x = translated_observation.x;
      observed_y = translated_observation.y;
      for (const auto &prediction : predictions) {
        if (prediction.id == translated_observation.id) {
          predicted_x = prediction.x;
          predicted_y = prediction.y;
        }
      }

      const double calculated_weight{
          calculate_weight(observed_x, observed_y, predicted_x, predicted_y,
                           observation_std_dev_x, observation_std_dev_y)};

      particle.weight *= calculated_weight;
    }
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> resampled_particles{};
  std::vector<double> weights{};
  std::default_random_engine random_engine{};

  double max_weight{0.}, beta{0.};
  for (const auto &particle : particles) {
    weights.emplace_back(particle.weight);
  }

  if (weights.size()) {
    max_weight = *std::max_element(weights.begin(), weights.end());
  }

  std::uniform_int_distribution<int> uniform_int_dist(0, num_particles);
  std::uniform_real_distribution<double> uniform_real_dist(0.0, max_weight);
  auto index{uniform_int_dist(random_engine)};

  for (const auto &particle : particles) {
    (void)particle;
    beta += uniform_real_dist(random_engine) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.emplace_back(particles[index]);
  }

  particles = std::move(resampled_particles);
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed
  // association sense_x: the associations x mapping already converted to
  // world coordinates sense_y: the associations y mapping already converted
  // to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
