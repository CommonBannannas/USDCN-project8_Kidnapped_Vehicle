/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 200;


  //Initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.x = x;
    p.y = y;
		p.id = i;
    p.theta = theta;
    p.weight = 1.0;

		//Normal distributions for sensor noise
	  normal_distribution<double> noise_x_(0, std[0]);
	  normal_distribution<double> noise_y_(0, std[1]);
	  normal_distribution<double> noise_theta_(0, std[2]);

    p.x += noise_x_(gen);
    p.y += noise_y_(gen);
    p.theta += noise_theta_(gen);

    particles.push_back(p);
  }

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  for (int i = 0; i < num_particles; i++) {

    // calculate new state, check div by zero
    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));

    }

		particles[i].theta += yaw_rate * delta_t;

		//Normal distributions for sensor noise
	  normal_distribution<double> noise_x(0, std_pos[0]);
	  normal_distribution<double> noise_y(0, std_pos[1]);
	  normal_distribution<double> noise_theta(0, std_pos[2]);

    particles[i].x += noise_x(gen);
    particles[i].y += noise_y(gen);
    particles[i].theta += noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) {

		//minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();
    //Id of landmark from map placeholder
    int map_id = -1;

		//Grab observation
    LandmarkObs o = observations[i];

    for (unsigned int j = 0; j < predicted.size(); j++) {
      //Grab prediction
      LandmarkObs p = predicted[j];

      //Distance between observed and predicted landmarks
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      //Find predicted landmark nearest the observed
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = p.id;
      }
    }

    observations[i].id = map_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	// for each particle...
 for (int i = 0; i < num_particles; i++) {

	 // get the particle x, y coordinates
	 double particle_x = particles[i].x;
	 double particle_y = particles[i].y;
	 double particle_theta = particles[i].theta;

	 vector<LandmarkObs> preds;

	 for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

		 float landmark_x = map_landmarks.landmark_list[j].x_f;
		 float landmark_y = map_landmarks.landmark_list[j].y_f;
		 int landmark_id = map_landmarks.landmark_list[j].id_i;

		 //check landmark in range of particles within a box
		 if (fabs(landmark_x - particle_x) <= sensor_range && fabs(landmark_y - particle_y) <= sensor_range) {

			 preds.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
		 }
	 }

	 //Transformed observations from vehicle coordinates to map coordinates
	 vector<LandmarkObs> trans_obs;
	 for (unsigned int j = 0; j < observations.size(); j++) {

		 double t_x = cos(particle_theta)*observations[j].x - sin(particle_theta)*observations[j].y + particle_x;
		 double t_y = sin(particle_theta)*observations[j].x + cos(particle_theta)*observations[j].y + particle_y;

		 trans_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });

	 }

	 dataAssociation(preds, trans_obs);

	 //Reinitialize weights
	 particles[i].weight = 1.0;

	 for (unsigned int j = 0; j < trans_obs.size(); j++) {

		 double obs_x, obs_y, pred_x, pred_y;
		 obs_x = trans_obs[j].x;
		 obs_y = trans_obs[j].y;

		 int associated_prediction = trans_obs[j].id;

		 for (unsigned int k = 0; k < preds.size(); k++) {
			 if (preds[k].id == associated_prediction) {
				 pred_x = preds[k].x;
				 pred_y = preds[k].y;
			 }
		 }

		 //Calcualte weight according to gaussian distribution.
		 double std_x = std_landmark[0];
		 double std_y = std_landmark[1];
		 double obs_weight = (1 / (2 * M_PI * std_x * std_y)) * exp(-(pow(pred_x - obs_x,2) / (2 * pow(std_x, 2)) + (pow(pred_y - obs_y,2)/(2 * pow(std_y, 2)))));

		 particles[i].weight *= obs_weight;
	 }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;

 // get all of the current weights
 vector<double> weights;
 for (int i = 0; i < num_particles; i++) {
	 weights.push_back(particles[i].weight);
 }

 //Resample
 uniform_int_distribution<int> uniintdist(0, num_particles-1);
 auto index = uniintdist(gen);

 // Max weight
 double max_weight = *max_element(weights.begin(), weights.end());


 uniform_real_distribution<double> unirealdist(0.0, max_weight);

 double beta = 0.0;

 for (int i = 0; i < num_particles; i++) {
	 beta += unirealdist(gen) * 2.0;
	 while (beta > weights[index]) {
		 beta -= weights[index];
		 index = (index + 1) % num_particles;
	 	}
	 new_particles.push_back(particles[index]);
 	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
